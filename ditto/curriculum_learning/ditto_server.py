import copy
import os
import sys

os.environ['RAY_DEDUP_LOGS'] = '0'

import shutil

import flwr as fl
from flwr.common import EvaluateRes, FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

from typing import List, Tuple, Union, Optional, Dict, Callable

from ditto_client import ditto_client_fn, ditto_client_fn_generator

from utils import *
from femnist import download_femnist


DEFAULT_CONFIG = {
    'local_epochs': 25,                                     # total epochs
    'loss_threshold': 95,                                   # depending on what you enter as loss type, this can be actual loss value or the percentile value you want to test for your scenario
    'test_name': 'test_percentile_hard',                    # put a meaningful test name
    'threshold_type': ThresholdType.PERCENTILE,             # change 0 for just flat num, 1, for percentile
    'percentile_type': 'linear',                            # change 'linear' for true percentile, 'normal_unbiased' for normal, put whatever for flat_num
    'curriculum_type': CurriculumType.TRANSFER_TEACHER,     # type of curriculum learning to use
    'lambda': None                                          # Ditto lambda value
}


def fit_config_fn_generator(config: Optional[dict] = None) -> Callable[[int], dict]:
    """
    Create a fit_config function to generate configurations for each round
    """

    if config is None:
        config = {}

    def fit_config(server_round: int) -> dict:
        """
        Return training configuration dict for each round.
        """
        round_config = copy.deepcopy(config)            # deep copy init_config to prevent aliasing
        round_config['server_round'] = server_round     # set current round

        return DEFAULT_CONFIG | round_config            # dictionary union, favoring config over DEFAULT_CONFIG

    return fit_config

# Default fit_config function
fit_config = fit_config_fn_generator()


# Modified from https://flower.ai/docs/framework/how-to-aggregate-evaluation-results.html
class DittoStrategy(fl.server.strategy.FedAvg):
    def __init__(self, log_accuracy=False, **kwargs):
        super().__init__(**kwargs)
        self.log_accuracy = log_accuracy


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
        weighted_global_accuracies = [r.metrics['global_accuracy'] * r.num_examples for _, r in results]
        weighted_local_accuracies = [r.metrics['local_accuracy'] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        avg_global_accuracy = sum(weighted_global_accuracies) / sum(examples)
        avg_local_accuracy = sum(weighted_local_accuracies) / sum(examples)
        if self.log_accuracy:
            print(f'Round {server_round} global accuracy aggregated from client results: {avg_global_accuracy}', file=sys.stderr)
            print(f'Round {server_round} local accuracy aggregated from client results: {avg_local_accuracy}', file=sys.stderr)


        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return \
            aggregated_loss, \
            {
                'avg_global_accuracy': avg_global_accuracy,
                'avg_local_accuracy': avg_local_accuracy,
                'local_loss': [r.metrics['local_loss'] for _, r in results],
                'global_accuracy': [r.metrics['global_accuracy'] for _, r in results],
                'local_accuracy': [r.metrics['local_accuracy'] for _, r in results]
            }



if __name__ == '__main__':
    # Download FEMNIST data (does nothing if already present)
    download_femnist()

    # Reset client models
    if os.path.exists(CLIENT_MODEL_DIR):
        shutil.rmtree(CLIENT_MODEL_DIR)  # Delete client model directory
    os.makedirs(CLIENT_MODEL_DIR)  # Recreate client model directory

    num_clients = 16

    # https://flower.ai/docs/framework/tutorial-series-customize-the-client-pytorch.html
    fl.simulation.start_simulation(
        num_clients=num_clients,
        client_fn=ditto_client_fn_generator(_lambda=1.0),
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=DittoStrategy(log_accuracy=True, on_fit_config_fn=fit_config),
        client_resources={
            'num_cpus': max(os.cpu_count()//num_clients, 1)
        }
    )
