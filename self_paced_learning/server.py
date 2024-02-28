import flwr as fl

# TODO: work on this fit_config function for more specialized cases
def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1, #if server_round < 2 else 2,  #
        "loss_threshold": 3 #if server_round < 2 else 2,  #

    }
    return config


SelfPaced = fl.server.strategy.FedAvg(
    on_fit_config_fn=fit_config        #fit_config,  # For future config function based changes
)

# Start Flower server (modified from https://flower.ai/)
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=SelfPaced
)