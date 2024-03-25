import flwr as fl
import ray
import gc
from client_femnist import client_fn
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateRes,
    FitRes,
    Scalar,
)
import numpy as np
import pandas as pd


TEST = "fem_sp_imgs_per_e"
ROUND = 50
EPOCHS = 25
NUM_CLIENTS = 8
THRESHOLD_TYPE = 1
PERCENTILE_TYPE = "linear"
LOSS_THRESHOLD = 4
test_lambda = [99.5]




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



results = {}


for _lambda in test_lambda:
    print('lambda =', _lambda)
    t_name = TEST +"_r"+ str(ROUND)+ "_e" + str(EPOCHS) + "_c" + str(NUM_CLIENTS) + "_l" +str(THRESHOLD_TYPE) + "_pt-" + str(PERCENTILE_TYPE) + "_lt" + str(_lambda)
    
    results = {}
    # TODO: work on this fit_config function for more specialized cases
    def fit_config(server_round: int):
        """Return training configuration dict for each round.

        Perform two rounds of training with one local epoch, increase to two local
        epochs afterwards.
        """
        config = {
            "server_round": server_round,           # The current round of federated learning
            "local_epochs": EPOCHS,                 # total epochs
            "loss_threshold": _lambda,       # depending on what you enter as loss type, this can be actual loss value
                                                    # or the percentile value you want to test for your scenario
            "Total_Rounds": ROUND,                  # total # of rounds
            "test_name": t_name,                 # put a meaningful test name
            "threshold_type": THRESHOLD_TYPE,       # change 0 for just flat num, 1, for percentile
            "percentile_type": PERCENTILE_TYPE      # change "linear" for true percentile, "normal_unbiased" for normal, put whatever for flat_num
        }
        return config

    # Create strategy and run server
    SelfPaced = AggregateCustomMetricStrategy(
        on_fit_config_fn=fit_config,        #fit_config,  # For future config function based changes
    )

    results[_lambda] = fl.simulation.start_simulation(
        num_clients=NUM_CLIENTS,
        clients_ids =["1","2","3","4","5","6","7","8"],
        client_fn=client_fn,
        config=fl.server.ServerConfig(num_rounds=ROUND),
        strategy=SelfPaced
    )
    losses_distributed = pd.DataFrame.from_dict({_lambda: [acc for _, acc in results[_lambda].losses_distributed] for _lambda in results.keys()})
    accuracies_distributed = pd.DataFrame.from_dict({_lambda: [acc for _, acc in results[_lambda].metrics_distributed['accuracy']] for _lambda in results.keys()})
    losses_distributed.to_csv('results/'+t_name+"/losses_distributed.csv")
    accuracies_distributed.to_csv('results/'+t_name+"/accuracies_distributed.csv")
    ray.shutdown()
    gc.collect()

