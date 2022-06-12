#!/bin/bash

### test
# sbatch train.sh "--augmentation=3 --h=10 --depth=4 --optimizer=Ranger21 --exp_name=testings --dry_run=True"

### v1 Baseline depth-based model
# sbatch train.sh "--epochs=25 --batch_size=512 --lr=0.0001 --h=512 --depth=16 --optimizer=AdamW --exp_name=v1-depth-baseline"
# sbatch train.sh "--epochs=25 --batch_size=512 --lr=0.0001 --h=512 --depth=16 --optimizer=Ranger21 --exp_name=v1-depth-baseline"
# sbatch train.sh "--epochs=25 --batch_size=512 --lr=0.0001 --h=512 --depth=16 --optimizer=Lamb --exp_name=v1-depth-baseline"
# sbatch train.sh "--epochs=25 --batch_size=512 --lr=0.0001 --h=512 --depth=16 --optimizer=Adam --exp_name=v1-depth-baseline"


### v2 Baseline hiddendim-based model
# sbatch train.sh "--epochs=25 --batch_size=512 --lr=0.0001 --h=960 --depth=8 --optimizer=AdamW --exp_name=v2-hiddendim_baseline"
# sbatch train.sh "--epochs=25 --batch_size=512 --lr=0.0001 --h=960 --depth=8 --optimizer=Ranger21 --exp_name=v2-hiddendim_baseline"
# sbatch train.sh "--epochs=25 --batch_size=512 --lr=0.0001 --h=960 --depth=8 --optimizer=Lamb --exp_name=v2-hiddendim_baseline"
# sbatch train.sh "--epochs=25 --batch_size=512 --lr=0.0001 --h=960 --depth=8 --optimizer=Adam --exp_name=v2-hiddendim_baseline"


### v3 Add data augmentation
# sbatch train.sh "--epochs=25 --batch_size=512 --lr=0.0001 --h=960 --depth=8 --optimizer=Ranger21 --augmentation=1 --exp_name=v3-augmentations"
# sbatch train.sh "--epochs=25 --batch_size=512 --lr=0.0001 --h=960 --depth=8 --optimizer=Ranger21 --augmentation=2 --exp_name=v3-augmentations"
# sbatch train.sh "--epochs=25 --batch_size=512 --lr=0.0001 --h=960 --depth=8 --optimizer=Ranger21 --augmentation=3 --exp_name=v3-augmentations"
# sbatch train.sh "--epochs=25 --batch_size=512 --lr=0.0001 --h=960 --depth=8 --optimizer=Ranger21 --augmentation=4 --exp_name=v3-augmentations"


### v4 based on v1/v2-models, decrease patch-size to 6, 5, 4 for larger interal-resolution and compare
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=0.0001 --h=960 --depth=8 --p_size=5 --optimizer=Ranger21 --exp_name=v4-patch_size"
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=0.0001 --h=512 --depth=16 --p_size=5 --optimizer=Ranger21 --exp_name=v4-patch_size"

# sbatch train.sh "--epochs=25 --batch_size=128 --lr=0.0001 --h=960 --depth=8 --p_size=4 --optimizer=Ranger21 --exp_name=v4-patch_size"
# sbatch train.sh "--epochs=25 --batch_size=128 --lr=0.0001 --h=512 --depth=16 --p_size=4 --optimizer=Ranger21 --exp_name=v4-patch_size"


### v5 based on v4 overfitting results, re-run on hiddendim models with lower LR to reduce overfit
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=0.00001 --h=960 --depth=8 --p_size=6 --optimizer=Ranger21 --exp_name=v5-p_size_low_LR"
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=0.00001 --h=960 --depth=8 --p_size=5 --optimizer=Ranger21 --exp_name=v5-p_size_low_LR"


### v6 vary kernel size with receptive field in mind, compare to v3 with aug=2 (current best model)
# sbatch train.sh "--epochs=25 --batch_size=512 --k_size=7 --lr=0.0001 --h=960 --depth=8 --optimizer=Ranger21 --augmentation=2 --exp_name=v6-k_sizes"
# sbatch train.sh "--epochs=25 --batch_size=512 --k_size=11 --lr=0.0001 --h=960 --depth=8 --optimizer=Ranger21 --augmentation=2 --exp_name=v6-k_sizes"


### v7 finetune AdamW for better training speed
# sbatch train.sh "--epochs=50 --batch_size=512 --lr=5e-6 --h=960 --depth=8 --optimizer=AdamW --augmentation=2 --decay=5e-2 --exp_name=v7-adamw_ftuning"
# sbatch train.sh "--epochs=50 --batch_size=512 --lr=1e-5 --h=960 --depth=8 --optimizer=AdamW --augmentation=2 --lr_policy=1CycleLR --exp_name=v7-adamw_ftuning"

