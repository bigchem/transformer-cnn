# Transformer-CNN

The repository contains the source code for a new Transformer-CNN method described in our paper https://chemrxiv.org/articles/Transformer-CNN_Fast_and_Reliable_Tool_for_QSAR/9961787. First, we trained the Transformer model on SMILES canonicalization task, e.g., given an arbitrary SMILES, the model converts it to a canonical one. Second, we use the internal representation of the Transformer (the output of the encoding stack with shape (BATCH, LENGTH, EMBEDDING)) as SMILES embeddings and build upon them CharNN model (Convolution and HighWay as it is done in DeepChem). The resulting model works both in classification and regression settings. 

Feel free to contact us if you have any suggestions or possible applications of this code.

# Dependencies 

The code has been tested in Ubuntu 18.04 with the following components:

1. python v.3.4.6 or higher
2. TensorFlow v1.12
3. rdkit v.2018.09.2

# How to use

The main program, transformer-cnn.py, uses the config.cfg file to read all the parameters of a task to do. After filling the config.cfg with the appropriate information, launch the python3 transformer-cnn.py config.cfg 

# How to train a model 

To train a model, one needs to create a config file like this. 
```
[Task]
   train_mode = True
   model_file = model.tar
   train_data_file = train.csv
[Details]
   canonize = True
   gpu = 0
   seed = 100666
   n_epochs = 30
   batch_size = 16
```
If canonize parameter is set then all the SMILES will be worked up with rdkit. Then 10 non-canonical SMILES for each molecule will be generated (the real number of generated strings can be smaller depending on the compound). If this parameter is set to False, then the string is passed to the model as is without any treatment. The same is also valid for the prognosis step. 

# Using the trained model 

To use a model, the config file looks like:
```
[Task]
   train_mode = False
   model_file = model.tar
   apply_data_file = predict.csv
[Details]
   canonize = True
   gpu = 0
   seed = 100666
   n_epochs = 30
   batch_size = 16
```
