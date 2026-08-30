import argparse
import logging
import multiprocessing as mp
import time

from src.baselines.fedavg.client import run_client
from src.baselines.fedavg.server import run_server
from src.data.data_loader import load_client_datasets
from src.utils import build_final_result_rows, parse_args, set_seed, write_csv_rows

logger = logging.getLogger(__name__)
formatter = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=formatter)


def main(argv=None):
    args = parse_args("fedavg", argv)
    set_seed(args.seed)

    datasets, _, active_clients = load_client_datasets(
        args.data_path,
        target_item=args.target_item,
        window_size=args.window_size,
        train_size=args.train_size,
        clients=args.clients,
    )
    client_ids = [cid for cid in active_clients if cid in datasets]

    server_address = f"{args.address}:{args.port}"
    manager = mp.Manager()
    result_rows = manager.list()

    server_process = mp.Process(
        target=run_server,
        args=(len(client_ids), args.port, args.fed_rounds),
    )
    server_process.start()
    time.sleep(3)

    client_processes = []
    for client_id in client_ids:
        client_args = argparse.Namespace(**vars(args))
        client_args.clients = [client_id]
        client_args.server_address = server_address
        process = mp.Process(target=run_client, args=(client_args, result_rows))
        process.start()
        client_processes.append(process)

    try:
        server_process.join()
        rows = list(result_rows)
        logger.info("fedavg_run rows_collected=%s", len(rows))
        if rows:
            final_rows = build_final_result_rows(
                rows,
                method="fedavg",
                target_item=args.target_item,
                members_by_client={cid: client_ids for cid in client_ids},
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
        if server_process.is_alive():
            server_process.terminate()
    manager.shutdown()


if __name__ == "__main__":
    main()
