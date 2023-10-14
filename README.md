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
	- `coil/coil.py` script to pre-process the COIL dataset (available [here](https://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php))
	- `amazon/amazon.py` script to pre-process the Amazon Picking Challenge dataset (available [here](https://rll.berkeley.edu/amazon_picking_challenge/))
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
To run the program, execute the main script `src/main_exec.py `. The script supports the following command line arguments:

```
main_exec.py [-h] -c <file> -g <num> [-f <frac>] [-l <model>] [-Ttrsaex]
```

- `-a`, `--accur` execute accuracy evaluation on selected samples (`-a`) or on all test set (`-aa`) *(it may take a while!)*.
- `-c <file>`, `--config <file>` pass a configuration file describing the model architecture and training parameters.
- `-e`, `--eval` execute evaluation routines.
- `-f <frac>`, `--fgpu <frac>` set the fraction of GPU memory to allocate *[default: 0.90]*.
- `-g <num>`, `--gpu <num>` set the number of GPUs to use (0 if CPU) or list of GPU indices.
- `-h`, `--help` show the help message with description of the arguments.
- `-i`, `--intrp` execute interpolation tests.
- `-l <model>`, `--load <model>` pass a folder or a HDF5 file to load as weights or entire model.
- `-p`, `--pred` compute model predictions over a selected set of images.
- `-r`, `--redir` redirect _stderr_ and _stdout_ to log files.
- `-s`, `--save` archive configuration file (`-s`) and python scripts (`-ss`) used.
- `-t`, `--test` execute testing routines.
- `-T`, `--train` execute training of the model.
- `-x`, `--hallx` execute hallucination routines.


As example, run the following command from the upmost `autoencoder/` folder. This command will train a new model on the first two GPUs on the machine. Then it will test the results, save all the files required to reproduce the experiment, and redirect all console messages to log files:

```
$ python main_exec.py -c config/cnfg_file -g 0,1 -Ttssr
```

Another example, this command will load an already trained model and will execute all the test routines on CPU:

```
$ python main_exec.py -l log/nicemodel/nn_best.h5 -c log/nicemodel/config/cnfg_file -g 0 -taaeipx
```
