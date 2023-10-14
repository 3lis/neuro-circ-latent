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
- `dataset/` structure of folders containing all data used to train the models.
- `src/` Python scripts:
	- `arch.py` defines the architectures of neural models,
	- `cnfg.py` handles command line arguments,
	- `exec_dset.py` creates the structures of symlinks for building a dataset,
	- `exec_eval.py` loads and evaluates a saved model,
	- `exec_feat.py` generates dataset of latent space encodings,
	- `exec_lata.py` is a collection of functions to analyze the latent space,
	- `exec_main.py` is the main file to execute training,
	- `gener.py` handles the *Generator* structures for parsing a dataset,
	- `h5lib.py` is a collection of utilities for loading weights from an HDF5 file,
	- `mesg.py` contains utilities for printing error messages,
	- `pred.py` defines a class for non-neural time prediction,
	- `sample_sel.py` contains a dictionary of manually-selected samples of different type of events,
	- `tester.py` collects functions for testing a trained model,
	- `trainer.py` contains the training routine.

## Usage
To run the program, execute the main script `src/exec_main.py `. The script supports the following command line arguments:

```
exec_main.py [-h] -c <file> -g <num> [-f <frac>] [-l <model>] [-Ttrsaex]
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
$ python src/exec_main.py -c config/cnfg_file -g 0,1 -Ttssr
```

Another example, this command will load an already trained model and will execute all the test routines on CPU:

```
$ python src/exec_main.py -l log/nicemodel/nn_best.h5 -c log/nicemodel/config/cnfg_file -g 0 -taaeipx
```
