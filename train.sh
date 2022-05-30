#!/bin/bash

#SBATCH -o /scratch/jakhac/ConvMixer/myjob.%j.%N.out   # Output-File
#SBATCH -D /scratch/jakhac/ConvMixer/ConvMixer-RS/src/    # Working Directory
#SBATCH --ntasks=2 		# Anzahl Prozesse P (CPU-Cores) 
#SBATCH --cpus-per-task=1	# Anzahl CPU-Cores pro Prozess P
#SBATCH --gres=gpu:tesla:2
#SBATCH --mem=100G              # 1GiB resident memory pro node

#SBATCH --time=12:00:00 # Erwartete Laufzeit
#SBATCH --partition=gpu

#Job-Status per Mail:
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hackstein@campus.tu-berlin.de

export https_proxy=http://frontend01:3128/ 

module load nvidia/cuda/11.2 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytest102

cd /scratch/jakhac/ConvMixer/ConvMixer-RS/src


python3 train.py $1
##python3 train.py --epochs=5 --batch_size=64 --lr=0.1 --h=128 --depth=2 --ds_size=200 --run_tests_n=3
