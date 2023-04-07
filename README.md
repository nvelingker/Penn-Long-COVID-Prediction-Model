# Penn's Honorable Mention Award Winning L3C Submission
This repository contains code that was originally written within the National Institute of Health's proprietary development environment; it has since been modified to run stand alone.  Within the codebase, you will find code for the machine learning models that were developed in a joint collaboration between the various engineering and medical departments at the University of Pennsylvania.  For a high level overview of our model, please [see here](https://www.challenge.gov/?challenge=l3c&tab=winners).

## Downloading the Data

The synthetic data can be found [here](https://drive.google.com/uc?export=download&id=1ov_PT9ScJFSZDJET3nKlrMyFbhIL4xV-).

## System Requirements

Required: learn how to install Spark [here](https://spark.apache.org/docs/latest/api/python/getting_started/install.html).

This codebase has been tested on a server with 2 Xeon 6284 CPUs, 767 GB of RAM, and 4 RTX 2080 Ti GPUs.

## Running the Code

1. Extract `synthetic_data.tar.gz` to the root directory.
```
$ tar -xvzf synthetic_data.tar.gz -C ./synthetic_data/
```
2. Create the folder `model_checkpoints` in the root directory.
```
$ mkdir model_checkpoints
```
3. Create a new conda environment using `requirements.txt` and activate it.
4. Train the models in the ensemble:
```
python train.py
```
5. Run the model on the synthetic testing data:
```
python infer.py
```
6. File `predictions.csv` will contain generated per-patient probablistic predictions.

## Acknowledgements

...