#!/bin/bash
#$ -V
#$ -j y
#$ -l h_rt=:10:
#$ -cwd
#$ -l arch=intel*
#$ -P gpu-training
#$ -l gpu=1

./reverse
