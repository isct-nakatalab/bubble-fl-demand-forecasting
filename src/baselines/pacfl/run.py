import argparse
import logging
import multiprocessing as mp
import time

from src.baselines.pacfl.client import run_client
from src.baselines.pacfl.clustering.clustering import identify_attackers, perform_clustering
from src.baselines.pacfl.clustering.differential_privacy import apply_differential_privacy
from src.baselines.pacfl.clustering.feature_importance import compute_all_feature_importances
from src.baselines.pacfl.server import run_server
from src.data.data_loader import load_client_datasets
from src.utils import build_final_result_rows, parse_args, set_seed, write_csv_rows

logger = logging.getLogger(__name__)
formatter = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=formatter)


def main(argv=None):
    args = parse_args("pacfl", argv)
    set_seed(args.seed)

    datasets, _, active_clients = load_client_datasets(
        args.data_path,
        target_item=args.target_item,
        window_size=args.window_size,
        train_size=args.train_size,
        clients=args.clients,
    )
    client_ids = [cid for cid in active_clients if cid in datasets]

    importance_matrix, client_ids = compute_all_feature_importances(datasets, client_ids)
    noisy_matrix = apply_differential_privacy(importance_matrix, epsilon=args.epsilon)
    _, _, bubbles = perform_clustering(noisy_matrix, args.n_clusters)
    _, safe_bubbles = identify_attackers(bubbles)

    manager = mp.Manager()
    result_rows = manager.list()
    members_by_client = {}

    server_processes = []
    client_processes = []
    try:
        for bubble_id, client_indices in safe_bubbles.items():
            port = args.port + bubble_id
            members = [client_ids[c] for c in client_indices]
            for client_id in members:
                members_by_client[client_id] = members
            server_address = f"{args.address}:{port}"

            server_process = mp.Process(target=run_server, args=(len(members), port, args.fed_rounds))
            server_process.start()
            server_processes.append(server_process)
            time.sleep(3)

            for client_id in members:
                client_args = argparse.Namespace(**vars(args))
                client_args.clients = [client_id]
                client_args.server_address = server_address
                client_args.bubble_id = bubble_id
                process = mp.Process(target=run_client, args=(client_args, result_rows))
                process.start()
                client_processes.append(process)

        for process in server_processes:
            process.join()
        rows = list(result_rows)
        logger.info("pacfl_run rows_collected=%s", len(rows))
        if rows:
            final_rows = build_final_result_rows(
                rows,
                method="pacfl",
                target_item=args.target_item,
                members_by_client=members_by_client,
            )
            write_csv_rows(args.out_path, list(final_rows[0].keys()), final_rows)
        for process in client_processes:
            process.join()
    except Exception as e:
        logger.warning(f"{e}")
    finally:
        for process in client_processes:
            if process.is_alive():
                process.terminate()
        for process in server_processes:
            if process.is_alive():
                process.terminate()

    manager.shutdown()


if __name__ == "__main__":
    main()
