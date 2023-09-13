#!/bin/sh

source /account/xchen/set_conda_env.sh

cd /account/xchen/workspace/AnimalGAN/github

conda activate AnimalGAN

python ./SRC/train.py --n_epochs 1000 > ./run/logs/Loss.txt 2>&1

conda deactivate
