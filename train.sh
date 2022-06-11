#!/bin/bash

#SBATCH -o /scratch/jakhac/ConvMixer/myjob.%j.%N.out   # Output-File
#SBATCH -D /scratch/jakhac/ConvMixer/src/    # Working Directory

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:tesla:2

## SBATCH --mem=100G          # 1GiB resident memory pro node
#SBATCH --mem-per-cpu=0

#SBATCH --cpus-per-task=1	# Anzahl CPU-Cores pro Prozess P
#SBATCH --time=48:00:00 # Erwartete Laufzeit
#SBATCH --partition=gpu

#Job-Status per Mail:
# #SBATCH --mail-type=NONE
# #SBATCH --mail-user=ddato@t-online.de

export https_proxy=http://frontend01:3128/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytest102

cd /scratch/jakhac/ConvMixer/src

echo "Args:"
echo $1

echo ""
echo "Start execution of train.py"
python3 train.py $1

