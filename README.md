## Project 2 - Classification on FashionMNIST using Multilayer Networks and CNNs

**Aim:** 

Train two models for classifying the FashionMNIST data. One model is a multi layer neural network, whereas the second model is a CNN based model.


**Dataset** 
Dataset used in this project is FashionMNIST.

**Folder description:**
- `utils.py` contains helper code used in other files
- `train_multilayer_nn.py` defines network architectures, trains, validates and saves the multilayer models. It also generates the required plots. Used for running multiple experiments.
- `CNN_models.py` defines many CNN network architectures used in this project.
- `train_conv_nn.py` trains, validates and saves the CNN models. It also generates the required plots. Used for running multiple experiments.
- `main.py` can be used to run inferences on new data. It loads the best models and writes 2 files `multi-layer-net.txt` (output from best multilayer network) and `convolution-neural-net.txt` (output from the best CNN model).
- `requirements.txt` contains the list of python packages used.
- `models` folder contains all the trained models
- `plots` folder contains plots of each experiment


**NOTE:** Run all the python files only from the current folder.