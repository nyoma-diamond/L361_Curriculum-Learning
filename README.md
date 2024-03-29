# Curriculum Learning Project

### Students: N'yoma Diamond, Grace Kim

## Notes for Reproducing the Self Paced Learning and Transfer Teacher Experiments:

Two main folders exist in the self-paced learning branch, the `self_paced_learning` and 	`transfer_teacher learning` client and server files.

For the `self_paced_learning` folder, two sets of code files exist for the FEMNIST and CIFAR-10 datasets. They are named: feminist_code, and CIFAR-10 code respectively. Each folder has their own `server.py`, `client.py`, and `utils.py` files, and experiments can be run by running `python server.py` in the correct directory in the users terminal. 

Each of the experiments have multiple configurations that can be changed directly in the `server.py` files, with the following variables that can be changed:

- `TEST` : [String] Name of the test being run, will be start of folder name save from results
- `ROUND` : [Int] Number of rounds being completed over the federated learning setting
- `EPOCHS` : [Int]  Number of Epochs being run for each round, FEMNIST needs a few more to raise accuracy levels compared to CIFAR-10. 
- `NUM_CLIENTS` : [Int] Number of clients being trained on in federated learning setting
- `THRESHOLD_TYPE` : [Int: 0 or 1] Set as 0 for just value based threshold, 1, for percentile
- `PERCENTILE_TYPE` : [String: “linear” or “true percentile”] Set as "linear" for true percentile, "normal_unbiased" for normal, does not matter what you put in for value based threshold
- `LOSS_THRESHOLD` : Can be ignored for large based runs
- `test_lambda` : [List, Int] Input a list of loss thresholds to test, examples for value based can be: [2,2.5,3,4], for percentile based (90 = 90%) can be [50,90,95] 

After setting the variables to whatever the user would like, results will be saved in the results folder of the directory being run in, with the folder being saved as the name of all of the variables that were run. Each of the experiments saved results folder will have the following files: 

- `accuracies_distributed.csv`
- `losses_distributed.csv`
- A folder for every client, named `cid_[put number of client ID here]`
    - And a folder for every round inside each client, each containing files for the round’s calculated losses named`losses.csv` and the failed images from that round

For the transfer-teacher learning folder, it is set up in a very similar format, with a folder for the FEMNIST dataset. However, it is advised to set the number of epochs to a higher number (10-25) to ensure that the ending effects/results of transfer teacher learning can be more evident.

