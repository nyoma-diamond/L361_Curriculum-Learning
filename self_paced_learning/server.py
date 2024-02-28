import flwr as fl

from client import client_fn



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

# https://flower.ai/docs/framework/tutorial-series-customize-the-client-pytorch.html
fl.simulation.start_simulation(
    num_clients=3,
    client_fn=client_fn,
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=SelfPaced
)