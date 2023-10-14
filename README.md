# Bio-inspired Circular Latent Spaces to Estimate Objects' Rotations

###### *Alice Plebe, Mauro Da Lio (2023).*
---

This repository contains the source code related to the paper *Bio-inspired Circular Latent Spaces to Estimate Objects' Rotations*.
<!--
The code is written in Keras 2.2.4 using TensorFlow 1.12.0 backend. The scripts are executed with Python 3.6.8. The networks are trained on multiple GPUs with CUDA 10.1.
The neural models obtained from Keras are exported to __Wolfram Mathematica 11.3__ for visualization.
-->

## Contents

- `README.md` this file.
- `dataset/` contains the folders with the training data:
	- `coil/coil.py` script to pre-process the COIL dataset (download available [here](https://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php))
	- `amazon/amazon.py` script to pre-process the Amazon Picking Challenge dataset (download available [here](https://rll.berkeley.edu/amazon_picking_challenge/))
- `src/` contains the main files:
	- `arch_net.py` defines the architectures of the neural models,
	- `cnfg_01.py` and `cnfg_02.py` are two examples of configuration files used to execute the program,
	- `data_gen.py` handles the generation of the dataset for training and testing,
 	- `img_utils.py` contains utility functions for image visualization,
	- `load_cnfg.py` loads the execution arguments passed from command line,     
	- `load_model.py` contains utility funtions for loading trained models,
   	- `main_exec.py` is the main file to execute the program,
   	- `print_msg.py` contains utility funtions for printing info during execution,
   	- `test_net.py` defines the testing routines to evaluate the models.

## Usage
To run the program, execute the main script `main_exec.py `. The script supports the following command line arguments:
- `-a`, `--angle` use a trained model in inverted mode to predict the angle,
- `-c <file>`, `--config <file>` pass a configuration file (without extension) describing the model architecture and training parameters,
- `-g <num>`, `--gpu <num>` set the number of GPUs to use (0 if CPU),
- `-i <nlist>`, `--index <nlist>` pass indexes of data samples to be tested with graphic output,
- `-l <model>`, `--load <model>` pass an HDF5 file to load as pre-trained model,
- `-L`, `--latent` execute latent analysis of the model,
- `-o <obj>`, `--object <obj>` pass the name of an object for which to show all shifted latents,
- `-r`, `--redir` redirect _stderr_ and _stdout_ to log files,
- `-s`, `--save` archive the configuration file and python scripts used,
- `-t`, `--test` execute testing routines,
- `-T`, `--train` execute training of the model.

For example, the following command trains a new model with architecture defined in the file `cnfg_01.py`. The execution uses one GPU, saves the sources files, and redirects all console messages to log files:

```
$ python main_exec.py -c cnfg_01 -g 1 -Trs
```

In another example, the following command trains a new model using the *InceptionV3* architecture as indicated in the file `cnfg_02.py`. The execution uses only the CPU, saves the sources files, and redirects all console messages to log files:

```
$ python main_exec.py -c cnfg_02 -g 0 -Trs
```
