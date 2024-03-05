import flwr as fl
from flwr.common import EvaluateRes, FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

from typing import List, Tuple, Union, Optional, Dict

from ditto_client import ditto_client_fn


# Modified from https://flower.ai/docs/framework/how-to-aggregate-evaluation-results.html
class DittoStrategy(fl.server.strategy.FedAvg):
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
        print(f'Round {server_round} global accuracy aggregated from client results: {avg_global_accuracy}')
        print(f'Round {server_round} local accuracy aggregated from client results: {avg_local_accuracy}')


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

# https://flower.ai/docs/framework/tutorial-series-customize-the-client-pytorch.html
fl.simulation.start_simulation(
    num_clients=5,
    client_fn=ditto_client_fn,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=DittoStrategy()
)