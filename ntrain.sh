#!/bin/bash

sbatch train.sh "--epochs=50 --batch_size=128 --lr=0.01 --h=256 --depth=6 --optimizer=Adam --exp_name=ddp_runtest --dry_run=True"

## DEPTH = {2, 4, 6}
# sbatch train.sh "--epochs=25 --batch_size=128 --lr=0.01 --h=64  --depth=2 --optimizer=Adam --exp_name=hparam_d_h_test"
# sbatch train.sh "--epochs=25 --batch_size=128 --lr=0.01 --h=128 --depth=2 --optimizer=Adam --exp_name=hparam_d_h_test"

# sbatch train.sh "--epochs=25 --batch_size=128 --lr=0.01 --h=128 --depth=4 --optimizer=Adam --exp_name=hparam_d_h_test"
# sbatch train.sh "--epochs=25 --batch_size=128 --lr=0.01 --h=256 --depth=4 --optimizer=Adam --exp_name=hparam_d_h_test"

# sbatch train.sh "--epochs=25 --batch_size=128 --lr=0.01 --h=256 --depth=6 --optimizer=Adam --exp_name=hparam_d_h_test"
# sbatch train.sh "--epochs=25 --batch_size=128 --lr=0.01 --h=512 --depth=6 --optimizer=Adam --exp_name=hparam_d_h_test"
