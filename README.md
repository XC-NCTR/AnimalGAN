# AnimalGAN_improved
This is an improved version, the original code is in the repository AnimalGAN_backup https://github.com/XC-NCTR/AnimalGAN_backup
# AnimalGAN: A Generative Adversarial Network Model Alternative to Animal Studies for Clinical Pathology Assessment
This repository provides the source codes for our paper **AnimalGAN: A Generative Adversarial Network Model Alternative to Animal Studies for Clinical Pathology Assessment**.

## Requirements
The code was tested with the packages listed in `environment.yml`. We assume that the installation of the above-mentioned packages covers all dependencies. In case we have missed essential dependencies please raise an issue. To allow you to reproduce our results easily, we also provide a Dockerfile that contains a working environment containing all the dependencies.
We provided an instruction on how to setup the required environment in the `Demo.mht`.

## Usage
The `main.py` is used to start (or resume) training, and generate clinical pathology measurements using a trained model.
```sh
bash ./scripts/run.sh 
```
Remember to set correct root path, data path, and checkpoint path.

### Hyperparameters
We have a module `opt.py` used to parse hyperparameters, the default values of the hyperparameters are provided. When running the `main.py` script, please specify the hyperparameters by using the `--[hyperparaname]` option.

### Training
All the hyperparameters regarding the training should be specified by using `--[hyperparaname] [value]`, the default values are also provided in the `opt.py`.
To train your own model, please use the `--train` option, and specify the hyperparameters you want to use.
```sh
python main.py --[hyperparaname] [value] --train
```
You can also use:
```sh
python train.py --[hyperparaname] [value]
```
More detais of usage are provided in the `Demo.mht`.

### Generating
To generate clinical pathology data for treatment conditions you are interested, please use the `--generate` option.
Please specify the number of valid records you want to generate using the `--num_generate` option with an integer.
```sh
python main.py --generate --num_generate 100 --model_path path/to/model --results_path where_to_save
```
You can also use:
```sh
python generate.py --num_generate 100 --model_path path/to/model --results_path where_to_save
```
More detais of usage are provided in the `Demo.mht`.
