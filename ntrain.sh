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

# sbatch train.sh "--epochs=25 --batch_size=256 --lr=0.0001 --h=960 --depth=8 --p_size=6 --optimizer=Ranger21 --exp_name=v4-patch_size"
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=0.0001 --h=512 --depth=16 --p_size=6 --optimizer=Ranger21 --exp_name=v4-patch_size"
