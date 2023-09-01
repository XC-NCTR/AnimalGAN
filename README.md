# AnimalGAN_improved
# AnimalGAN: A Generative Adversarial Network Model Alternative to Animal Studies for Clinical Pathology Assessment
This repository provides the source codes for our paper **AnimalGAN: A Generative Adversarial Network Model Alternative to Animal Studies for Clinical Pathology Assessment**.

## Requirements
The code was tested with the packages listed in requirements.txt. We assume that the installation of the above-mentioned packages covers all dependencies. In case we have missed essential dependencies please raise an issue. To allow you to reproduce our results easily, we also provide a Dockerfile that contains a working environment containing all the dependencies.

## Usage
The `main.py` is used to start (or resume) training, and generate clinical pathology measurements using a trained model.
```sh
run ./scripts/run_train.sh 
```
Remember to set correct root path, data path, and checkpoint path.

### Hyperparameters
Hyperparameters are defined in a `hyperparameters.json` file.
It contains the path to the directory where the data, models, logs and results will be stored, along with a list of GPU identifiers (0..n) to be used, given that each model will be trained on a single GPU.
When running the `main.py` script, please specify the hyperparameters by using the `--hyperpara` option with the path to the hyperparameters defined in a `.json` file.

### Training
All the hyperparameters regarding the training should be defined in the `.json` file.
The size of the different layers, the normalization layer used to condition the generation (batchnorm or layernorm), lambda, the batch size, dimension of the latent space, along with all the optimizer parameters (number of steps, learning rate, algorithm used...) are defined there.
Also, the frequencies (how often the values are logged, the t-SNE validation plots are plotted, the loss values are displayed on the standard output, the model is saved) are defined there.
To train your own model, please use the `--train` option, and specify the hyperparameters you want to use in the `.json` file.
```sh
python main.py --hyperparam hyperparameters.json --train
```
Note that you can also resume the training by specifying the path to a checkpoint in the `.json` file.

### Generating
To generate clinical pathology data for treatment conditions you are interested, please use the `--generate` option.
Please specify the number of valid records you want to generate using the `--num` option with an integer.
```sh
python main.py --hyperparam hyperparameters.json --generate --num 100 --model_path path/to/my/model --save_path where_to_save.h5ad
```
