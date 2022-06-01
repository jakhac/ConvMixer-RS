#!/bin/bash

#SBATCH -o /scratch/jakhac/ConvMixer/myjob.%j.%N.out   # Output-File
#SBATCH -D /scratch/jakhac/ConvMixer/src/    # Working Directory

##SBATCH --ntasks=2 		# Anzahl Prozesse P (CPU-Cores) 
#SBATCH --cpus-per-task=1	# Anzahl CPU-Cores pro Prozess P

#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:tesla:2
#SBATCH --mem=100G              # 1GiB resident memory pro node

#SBATCH --time=00:05:00 # Erwartete Laufzeit
#SBATCH --partition=gpu_short

#Job-Status per Mail:
# #SBATCH --mail-type=NONE
# #SBATCH --mail-user=ddato@t-online.de

export https_proxy=http://frontend01:3128/ 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytest102

export MASTER_PORT=12340
export WORLD_SIZE=6

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# export NCCL_DEBUG=INFO


cd /scratch/jakhac/ConvMixer/src

echo "Args:"
echo $1

echo ""
echo "Start execution of train.py"

srun python3 train.py $1
