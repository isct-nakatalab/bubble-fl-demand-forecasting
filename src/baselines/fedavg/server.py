import flwr as fl


def weighted_average(metrics):
    examples = [n for n, _ in metrics]
    total = sum(examples)
    result = {}
    for key in ["rmse", "r_squared", "mse", "mae", "loss"]:
        values = [n * m[key] for n, m in metrics if key in m]
        if values:
            result[key] = sum(values) / total
    return result


def run_server(num_clients: int, port: int = 8080, fed_rounds: int = 100):
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=fed_rounds),
        strategy=strategy,
    )
