# Curriculum Learning Project

### Students: N'yoma Diamond, Grace Kim

Required python modules are specified in `requirements.txt`.

## Notes for Reproducing the Self Paced Learning and Transfer Teacher Experiments:

Two main folders exist in the self-paced learning branch, the `self_paced_learning` and 	`transfer_teacher learning` client and server files.

For the `self_paced_learning` folder, two sets of code files exist for the FEMNIST and CIFAR-10 datasets. They are named: feminist_code, and CIFAR-10 code respectively. Each folder has their own `server.py`, `client.py`, and `utils.py` files, and experiments can be run by running `python server.py` in the correct directory in the users terminal. 

Each of the experiments have multiple configurations that can be changed directly in the `server.py` files, with the following variables that can be changed:

- `TEST`: [String] Name of the test being run, will be start of folder name save from results
- `ROUND`: [Int] Number of rounds being completed over the federated learning setting
- `EPOCHS`: [Int]  Number of Epochs being run for each round, FEMNIST needs a few more to raise accuracy levels compared to CIFAR-10. 
- `NUM_CLIENTS`: [Int] Number of clients being trained on in federated learning setting
- `THRESHOLD_TYPE`: [Int: 0 or 1] Set as 0 for just value based threshold, 1, for percentile
- `PERCENTILE_TYPE`: [String: “linear” or “true percentile”] Set as "linear" for true percentile, "normal_unbiased" for normal, does not matter what you put in for value based threshold
- `LOSS_THRESHOLD`: Can be ignored for large based runs
- `test_lambda`: [List, Int] Input a list of loss thresholds to test, examples for value based can be: [2,2.5,3,4], for percentile based (90 = 90%) can be [50,90,95] 

After setting the variables to whatever the user would like, results will be saved in the results folder of the directory being run in, with the folder being saved as the name of all of the variables that were run. Each of the experiments saved results folder will have the following files: 

- `accuracies_distributed.csv`
- `losses_distributed.csv`
- A folder for every client, named `cid_[put number of client ID here]`
    - And a folder for every round inside each client, each containing files for the round’s calculated losses named`losses.csv` and the failed images from that round

For the transfer-teacher learning folder, it is set up in a very similar format, with a folder for the FEMNIST dataset. However, it is advised to set the number of epochs to a higher number (10-25) to ensure that the ending effects/results of transfer teacher learning can be more evident.

## Notes for Reproducing the `Ditto` Experiments:

The code for `Ditto` is present in the `ditto` folder, containing two files: `ditto_client.py` and `ditto_server.py`. All `Ditto` experiments are run on FEMNIST. Note that these files rely heavily on code present in the `utils.py` file in the root directory of the project.
 
Experiment configurations for `Ditto` can be configured using the `fit_config_fn_generator` function in `ditto_server.py`. This function can have the following (all optional) parameters provided (in dictionary form):

 - `local_epochs`: [Int] Number of Epochs being run for each round. Defaults to 25.
 - `loss_threshold`: [Float] The loss cutoff threshold used by self-paced/transfer-teacher learning if enabled. Defaults to 0.95.
 - `threshold_type`: [ThresholdType] The type of threshold provided (e.g. percentile, quantile, or direct value). Defaults to ThresholdType.QUANTILE.
 - `percentile_type`: [String] Passed to the `method` argument of `numpy.percentile` or `numpy.quantile` if percentile or quantile methods are used. Defaults to 'linear'.
 - `curriculum_type`: [CurriculumType] The type of curriculum learning method to use (none, self-paced, or transfer-teacher). Defaults to CurriculumType.TRANSFER_TEACHER.
 - `lambda`: [Float] Lambda bias value to be used by `ditto`. Defaults to 1.0.

Code Example code for running experiments, and the code that was used to run the `ditto` experiments used in our report are present in the `experiments.ipynb` Jupyter notebook.

Note: `Ditto` experiments create and save copies of the client models in a temporary folder called `client_models`. This is used to store and maintain each `Ditto` client's personalized local model on disk. The experiments are set up to automatically delete this folder before each experiment. If you wish to run multiple experiments simultaneously, the code will need to be modified so each experiment saves their client models to different locations to prevent experiments overwriting each others' models. 