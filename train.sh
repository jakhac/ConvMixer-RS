#!/bin/bash

#SBATCH -o /scratch/jakhac/ConvMixer/myjob.%j.%N.out   # Output-File
#SBATCH -D /scratch/jakhac/ConvMixer/src/    # Working Directory

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1

## SBATCH --mem=100G          # 1GiB resident memory pro node
#SBATCH --mem-per-cpu=0

#SBATCH --cpus-per-task=1	# Anzahl CPU-Cores pro Prozess P
#SBATCH --time=00:10:00 # Erwartete Laufzeit
#SBATCH --partition=gpu

#Job-Status per Mail:
# #SBATCH --mail-type=NONE
# #SBATCH --mail-user=ddato@t-online.de

export https_proxy=http://frontend01:3128/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytest


NB_GPUS=$(nvidia-smi -L | grep "UUID: MIG-GPU" | wc -l)
if [[ "$NB_GPUS" == 0 ]]; then
    ALL_GPUS=$(nvidia-smi -L | grep "UUID: GPU" | cut -d" " -f5 | cut -d')' -f1)

    echo "No MIG GPU available, using the full GPUs ($ALL_GPUS)."
else
    ALL_GPUS=$(nvidia-smi -L | grep "UUID: MIG-GPU" | cut -d" " -f8 | cut -d')' -f1)
    echo "Found $NB_GPU MIG instances: $ALL_GPUS"
fi

for gpu in $(echo "$ALL_GPUS"); do
    export CUDA_VISIBLE_DEVICES=$gpu
done

echo "CUDA_VISIBLE_DEVICES: " $CUDA_VISIBLE_DEVICES

cd /scratch/jakhac/ConvMixer/src

echo "Args:"
echo $1

echo ""
echo "Start execution of train.py"
python3 train.py $1

