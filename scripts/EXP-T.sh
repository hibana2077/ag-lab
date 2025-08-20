#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12
#PBS -l mem=8GB           
#PBS -l walltime=05:30:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/ag-lab/.venv/bin/activate

cd ..
python3 benchmark.py --task needle --model transformer --context_len 256 --epochs 300 >> T001.log
python3 benchmark.py --task needle --model transformer --context_len 512 --epochs 300 >> T002.log
python3 benchmark.py --task needle --model transformer --context_len 1024 --epochs 300 >> T003.log