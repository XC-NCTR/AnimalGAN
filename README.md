# AnimalGAN: A Generative Adversarial Network Model Alternative to Animal Studies for Clinical Pathology Assessment
This repository provides the source codes for our paper **AnimalGAN: A Generative Adversarial Network Model Alternative to Animal Studies for Clinical Pathology Assessment**.

## Repository Structure
    ├── Data/                                # Directory for example data
    │   ├── SDFs/                            # Structure-Data Files (SDFs) of compounds of interest
    │   ├── Example_Data_training.tsv        # Example of the training dataset
    │   └── Example_MolecularDescriptors.tsv # Example of the molecular descriptors
    ├── SRC/                                 # Directory for source code
    │   ├── model.py                         # Define the Generator class and the Discriminator class
    │   ├── train_cwgangp.py                 # Script for training the model using Conditional WGAN with gradient penalty (CWGAN_GP)
    │   ├── train_cwgangp_scale.py           # Script for training the model using CWGAN_GP with different scaling
    │   ├── train.py                         # Script for training the model
    │   ├── generate.py                      # Script for generating data using the pretrained model
    │   └── utils.py                         # Utility functions
    └── environment.yml                      # Environment configuration file

## Requirements
The code was tested with the packages listed in `environment.yml`. We assume that the installation of the above-mentioned packages covers all dependencies. In case we have missed essential dependencies please raise an issue. To allow you to reproduce our results easily, we provided an instruction on how to setup the required environment and run the code in the `Demo.md`.

## Usage
```sh
bash ./scripts/run.sh 
```
Remember to set correct root path, data path, and checkpoint path.

### Hyperparameters
We have a module `opt.py` used to parse hyperparameters, the default values of the hyperparameters are provided. When running the script, please specify the hyperparameters by using the `--[hyperparaname]` option.

### Training
All the hyperparameters regarding the training should be specified by using `--[hyperparaname] [value]`, the default values are also provided in the `opt.py`.
To train your own model, please specify the hyperparameters you want to use.
```sh
python train.py --[hyperparaname] [value]
```
More detais of usage are provided in the `Demo.md`.

### Generating
To generate clinical pathology data for treatment conditions you are interested, please specify the number of valid records you want to generate using the `--num_generate` option with an integer.
```sh
python generate.py --num_generate 100 --model_path path/to/model --results_path where_to_save
```
More detais of usage are provided in the `Demo.md`.
