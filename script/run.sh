#!/bin/sh
#
#$ -N iDILI
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l h_vmem=20G
#$ -pe smp 12
#$ -R y
#$ -q hpc.q@ncshpcgpu01.fda.gov
#$ -w e
#$ -o /account/xchen/workspace/AnimalGAN/github/run/train.log
#$ -e /account/xchen/workspace/AnimalGAN/github/run/train.err

source /account/xchen/set_conda_env.sh

cd /account/xchen/workspace/AnimalGAN/github

conda activate AnimalGAN

python ./SRC/train.py --n_epochs 1000 > ./run/logs/Loss.txt 2>&1

conda deactivate