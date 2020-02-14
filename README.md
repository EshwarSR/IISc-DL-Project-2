## Project 1 - FizzBuzz

**Aim:** 

Given a number, model should output
- `fizz` if number divisible by 3
- `buzz` if number divisible by 5
- `fizzbuzz` if number divisible by 15
- `number itself` otherwise

**Training data:** 

Numbers from 101 to 1000

**Test data:** 

Numbers from 1 to 100

**Folder description:**
- `utils.py` contains helper code used in other files
- `create_dataset.py` creates the `training_data.csv` and `test_data.csv` files, useful for traing the models.
- `train.py` defines network architectures, trains, tests and saves the models. It also generates the required plots. Used for running multiple experiments.
- `main.py` can be used to run inferences on new data. It loads the best model and writes 2 files `Software1.txt` (rule based classification) and `Software2.txt` (model based classification).
- `requirements.txt` contains the list of python packages used.
- `model` folder contains all the trained models
- `plots` folder contains plots of each experiment


**NOTE:** Run all the python files only from the current folder.