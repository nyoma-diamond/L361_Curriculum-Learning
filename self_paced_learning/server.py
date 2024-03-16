import flwr as fl

from client import client_fn
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateRes,
    FitRes,
    Scalar,
)

ROUND = 2
EPOCHS = 2
LOSS_THRESHOLD = 95
TEST_NAME = "test_percentile_hard"
THRESHOLD_TYPE = 1
PERCENTILE_TYPE = "linear"

# TODO: work on this fit_config function for more specialized cases
def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,           # The current round of federated learning
        "local_epochs": EPOCHS,                 # total epochs
        "loss_threshold": LOSS_THRESHOLD,       # depending on what you enter as loss type, this can be actual loss value
                                                # or the percentile value you want to test for your scenario
        "Total_Rounds": ROUND,                  # total # of rounds
        "test_name": TEST_NAME,                 # put a meaningful test name
        "threshold_type": THRESHOLD_TYPE,       # change 0 for just flat num, 1, for percentile
        "percentile_type": PERCENTILE_TYPE      # change "linear" for true percentile, "normal_unbiased" for normal, put whatever for flat_num
    }
    return config


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}

# SelfPaced = fl.server.strategy.FedAvg(
#     on_fit_config_fn=fit_config        #fit_config,  # For future config function based changes
# )

    
# Create strategy and run server
SelfPaced = AggregateCustomMetricStrategy(
    on_fit_config_fn=fit_config,        #fit_config,  # For future config function based changes
)
# https://flower.ai/docs/framework/tutorial-series-customize-the-client-pytorch.html
fl.simulation.start_simulation(
    num_clients=3,
    client_fn=client_fn,
    config=fl.server.ServerConfig(num_rounds=ROUND),
    strategy=SelfPaced
)