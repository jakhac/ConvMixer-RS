#!/bin/bash

### test
# sbatch train.sh "--augmentation=2 --h=10 --depth=4 --optimizer=AdamW --exp_name=testings --dry_run=True"

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
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=0.0001 --h=960 --depth=8 --p_size=5 --optimizer=Ranger21 --exp_name=v4-psize"
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=0.0001 --h=512 --depth=16 --p_size=5 --optimizer=Ranger21 --exp_name=v4-psize"
# sbatch train.sh "--epochs=25 --batch_size=128 --lr=0.0001 --h=960 --depth=8 --p_size=4 --optimizer=Ranger21 --exp_name=v4-psize"
# sbatch train.sh "--epochs=25 --batch_size=128 --lr=0.0001 --h=512 --depth=16 --p_size=4 --optimizer=Ranger21 --exp_name=v4-psize"


### v5 based on v4 overfitting results, re-run on hiddendim models with lower LR to reduce overfit
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=0.00001 --h=960 --depth=8 --p_size=6 --optimizer=Ranger21 --exp_name=v5-psize_low_LR"
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=0.00001 --h=960 --depth=8 --p_size=5 --optimizer=Ranger21 --exp_name=v5-psize_low_LR"


### v6 vary kernel size with receptive field in mind, compare to v3 with aug=2 (current best model)
# sbatch train.sh "--epochs=25 --batch_size=512 --k_size=7 --lr=0.0001 --h=960 --depth=8 --optimizer=Ranger21 --augmentation=2 --exp_name=v6-k_sizes"
# sbatch train.sh "--epochs=25 --batch_size=512 --k_size=11 --lr=0.0001 --h=960 --depth=8 --optimizer=Ranger21 --augmentation=2 --exp_name=v6-k_sizes"
# sbatch train.sh "--epochs=25 --batch_size=512 --k_size=5 --lr=0.0001 --h=960 --depth=8 --optimizer=Ranger21 --augmentation=2 --exp_name=v6-k_sizes"
# sbatch train.sh "--epochs=25 --batch_size=512 --k_size=13 --lr=0.0001 --h=960 --depth=8 --optimizer=Ranger21 --augmentation=2 --exp_name=v6-k_sizes"


### v7 finetune AdamW for better training speed
# sbatch train.sh "--epochs=50 --batch_size=512 --lr=5e-6 --h=960 --depth=8 --optimizer=AdamW --augmentation=2 --decay=5e-2 --exp_name=v7-adamw_ftuning"
# sbatch train.sh "--epochs=50 --batch_size=512 --lr=1e-5 --h=960 --depth=8 --optimizer=AdamW --augmentation=2 --lr_policy=1CycleLR --exp_name=v7-adamw_ftuning"


### v8 based on v5, still potential in lower patch_sizes, add aug and AdamW instead of Ranger21 (training speed ..)
# sbatch train.sh "--epochs=50 --batch_size=256 --lr=1e-5 --h=960 --depth=8 --p_size=5 --optimizer=AdamW --augmentation=2 --decay=5e-2 --exp_name=v5-psize_optims"
# sbatch train.sh "--epochs=50 --batch_size=256 --lr=5e-6 --lr_policy=RLROP --h=960 --depth=8 --p_size=5 --optimizer=AdamW --augmentation=2 --decay=5e-2 --exp_name=v5-psize_optims"
# sbatch train.sh "--epochs=50 --batch_size=256 --lr=1e-4 --h=960 --depth=8 --p_size=5 --optimizer=Ranger21 --augmentation=2 --exp_name=v5-psize_optims"


### v9 based on v8 (p_size=5, Ranger21 LR=1e-4) and v6 (k_size=5/7)
# sbatch train.sh "--epochs=35 --batch_size=256 --lr=1e-4 --h=960 --depth=8 --p_size=5 --k_size=5 --optimizer=Ranger21 --augmentation=2 --exp_name=v9-pk_sizes"
# sbatch train.sh "--epochs=35 --batch_size=256 --lr=1e-4 --h=960 --depth=8 --p_size=5 --k_size=7 --optimizer=Ranger21 --augmentation=2 --exp_name=v9-pk_sizes"


### v10 Hdim adjustement
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=1e-4 --h=1024 --depth=8 --p_size=5 --k_size=5 --optimizer=Ranger21 --augmentation=2 --exp_name=v10-hdim-inc"


### v11 Residual adjustement
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=1e-4 --h=960 --depth=8 --p_size=5 --k_size=5 --optimizer=Ranger21 --augmentation=2 --residual=0 --exp_name=v11-residual"
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=1e-4 --h=960 --depth=8 --p_size=5 --k_size=5 --optimizer=Ranger21 --augmentation=2 --residual=2 --exp_name=v11-residual"
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=1e-4 --h=960 --depth=8 --p_size=5 --k_size=5 --optimizer=Ranger21 --augmentation=2 --residual=3 --exp_name=v11-residual"


### v12 dropouts
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=1e-4 --h=768 --depth=8 --p_size=5 --k_size=5 --optimizer=Ranger21 --augmentation=2 --drop=0.1 --exp_name=v12-dropouts"
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=1e-4 --h=768 --depth=8 --p_size=5 --k_size=5 --optimizer=Ranger21 --augmentation=2 --drop=0.25 --exp_name=v12-dropouts"


### v13 Activation ReLU
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=1e-4 --h=1024 --depth=8 --p_size=5 --k_size=5 --optimizer=Ranger21 --augmentation=2 --activation=ReLU --exp_name=v13-relu"


### v14 Reproduce v9 results with AdamW
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=1e-4 --h=960 --depth=8 --p_size=5 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --augmentation=2 --exp_name=v14-AdamW"
# sbatch train.sh "--epochs=40 --batch_size=256 --lr=1e-4 --h=960 --depth=8 --p_size=5 --k_size=5 --optimizer=AdamW --lr_warmup_fn=exp --augmentation=2 --exp_name=v14-AdamW"


### v15 Finetune AdamW with ReLU and h=1024
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --h=1024 --depth=8 --p_size=5 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v15-AdamW_relu"
# sbatch train.sh "--epochs=35 --batch_size=256 --lr=5e-5 --h=1024 --depth=8 --p_size=5 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v15-AdamW_relu"


### v16 Can deeper ConvMixers compensate a small hidden dimension?
#--- h1024 (benchmark for this experiment)
# sota v15 comp-> "--epochs=25 --batch_size=256 --lr=1e-4 --h=1024 --depth=8  --p_size=5 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v15-AdamW_relu"
#--- h960
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --h=960  --depth=8  --p_size=5 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v16-deep_cvmx"
# sbatch train.sh "--epochs=30 --batch_size=128 --lr=1e-4 --h=960  --depth=16 --p_size=5 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v16-deep_cvmx"
#--- h512
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --h=512  --depth=8  --p_size=5 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v16-deep_cvmx"
# sbatch train.sh "--epochs=30 --batch_size=256 --lr=1e-4 --h=512  --depth=16 --p_size=5 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v16-deep_cvmx"


### v17 Can deeper ConvMixers compensate low internal resolution?
#--- p=5 benchmark (benchmark for this experiment / comparison)
# sbatch train.sh "--epochs=30 --batch_size=256 --lr=1e-4 --h=512 --depth=8 --p_size=5 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v17-depth_psize"
#--- p=7 v17-.*p=5|v17-.*p=7
# sbatch train.sh "--epochs=30 --batch_size=256 --lr=1e-4 --h=512 --depth=8  --p_size=7 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v17-depth_psize"
# sbatch train.sh "--epochs=30 --batch_size=256 --lr=1e-4 --h=512 --depth=12 --p_size=7 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v17-depth_psize"
# sbatch train.sh "--epochs=30 --batch_size=256 --lr=1e-4 --h=512 --depth=16 --p_size=7 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v17-depth_psize"
# sbatch train.sh "--epochs=30 --batch_size=256 --lr=1e-4 --h=512 --depth=20 --p_size=7 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v17-depth_psize"
# sbatch train.sh "--epochs=35 --batch_size=256 --lr=1e-4 --h=512 --depth=24 --p_size=7 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v17-depth_psize"
#--- p=9
# sbatch train.sh "--epochs=30 --batch_size=256 --lr=1e-4 --h=512 --depth=8  --p_size=9 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v17-depth_psize"
# sbatch train.sh "--epochs=30 --batch_size=256 --lr=1e-4 --h=512 --depth=16 --p_size=9 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v17-depth_psize"
# sbatch train.sh "--epochs=35 --batch_size=256 --lr=1e-4 --h=512 --depth=24 --p_size=9 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v17-depth_psize"


### v18 Add dilation to best performing model (first of v17)
## v18|v16.*h=512-d=8
# sbatch train.sh "--epochs=30 --batch_size=256 --lr=1e-4 --h=512 --depth=8 --p_size=5 --k_size=5 --k_dilation=2 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v18-dilation"


### v19 Weight decay (default is 1e-2)
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --h=960  --depth=8  --p_size=5 --k_size=5 --optimizer=AdamW --decay=1e-1 --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v19-wdecay"
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --h=960  --depth=8  --p_size=5 --k_size=5 --optimizer=AdamW --decay=1e-3 --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v19-wdecay"
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --h=960  --depth=8  --p_size=5 --k_size=5 --optimizer=AdamW --decay=1e-4 --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v19-wdecay"


### v20 Repeat best runs from v15 / v16
# sbatch train.sh "--epochs=30 --batch_size=256 --lr=1e-4 --h=512  --depth=16 --p_size=5 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v20-normalized"
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --h=1024 --depth=8 --p_size=5 --k_size=5 --optimizer=AdamW --lr_warmup_fn=linear --activation=ReLU --augmentation=2 --exp_name=v20-normalized"


##### Comparison Models #####

### r1 - ResNet-50
# sbatch train.sh "--epochs=35 --batch_size=1024 --lr=1e-4 --optimizer=AdamW --lr_schedule=0 --arch=ResNet50 --augmentation=2 --exp_name=r1-baseline"
# sbatch train.sh "--epochs=35 --batch_size=1024 --lr=1e-4 --optimizer=AdamW --lr_warmup_fn=linear --arch=ResNet50 --augmentation=2 --exp_name=r1-baseline"

### r2 - Rerun with normalized data
# sbatch train.sh "--epochs=35 --batch_size=1024 --lr=1e-4 --optimizer=AdamW --lr_warmup_fn=linear --arch=ResNet50 --augmentation=2 --exp_name=r2-normalized"
# sbatch train.sh "--epochs=35 --batch_size=1024 --lr=1e-4 --optimizer=AdamW --lr_warmup_fn=linear --arch=ResNet18 --augmentation=2 --exp_name=r2-normalized"




### t1 VisionTransformer
# sbatch train.sh "--epochs=30 --batch_size=512 --lr=1e-4 --optimizer=AdamW --lr_warmup_fn=linear --arch=ViT --augmentation=2 --exp_name=t1-baseline"
# sbatch train.sh "--epochs=30 --batch_size=512 --lr=1e-5 --optimizer=AdamW --lr_warmup_fn=linear --arch=ViT --augmentation=2 --exp_name=t1-baseline"

### t2 Recommended hparams
# sbatch train.sh "--epochs=30 --batch_size=1024 --lr=1e-4 --depth=8 --num_heads=8 --embed_dim=256 --p_size=20 --drop=0.2 --attn_drop=0.2 --optimizer=AdamW --lr_warmup_fn=linear --arch=ViT --augmentation=2 --exp_name=t2-hparams"
# sbatch train.sh "--epochs=30 --batch_size=1024 --lr=1e-4 --depth=12 --num_heads=8 --embed_dim=256 --p_size=20 --drop=0.2 --attn_drop=0.2 --optimizer=AdamW --lr_warmup_fn=linear --arch=ViT --augmentation=2 --exp_name=t2-hparams"
# sbatch train.sh "--epochs=30 --batch_size=1024 --lr=1e-4 --depth=8 --num_heads=8 --embed_dim=256 --p_size=16 --drop=0.2 --attn_drop=0.2 --optimizer=AdamW --lr_warmup_fn=linear --arch=ViT --augmentation=2 --exp_name=t2-hparams"

### t3 Change optimizing
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --depth=8 --num_heads=8 --embed_dim=256 --p_size=20 --drop=0.2 --attn_drop=0.2 --path_drop=0.0 --optimizer=AdamW --lr_warmup_fn=linear --arch=ViT --augmentation=2 --exp_name=t3-batchsize"
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --depth=8 --num_heads=8 --embed_dim=256 --p_size=20 --drop=0.2 --attn_drop=0.2 --path_drop=0.2 --optimizer=AdamW --lr_warmup_fn=linear --arch=ViT --augmentation=2 --exp_name=t3-batchsize"
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --depth=8 --num_heads=8 --embed_dim=256 --p_size=20 --drop=0.0 --attn_drop=0.0 --path_drop=0.0 --optimizer=AdamW --lr_warmup_fn=linear --arch=ViT --augmentation=2 --exp_name=t3-batchsize"

### t4 Change optimizing
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --depth=12 --num_heads=8 --embed_dim=256 --p_size=20 --drop=0.0 --attn_drop=0.0 --path_drop=0.0 --optimizer=AdamW --lr_warmup_fn=linear --arch=ViT --augmentation=2 --exp_name=t4-arch"
# sbatch train.sh "--epochs=25 --batch_size=128 --lr=1e-4 --depth=12 --num_heads=8 --embed_dim=256 --p_size=20 --drop=0.0 --attn_drop=0.0 --path_drop=0.0 --optimizer=AdamW --lr_warmup_fn=linear --arch=ViT --augmentation=2 --exp_name=t4-arch"
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-5 --depth=12 --num_heads=8 --embed_dim=256 --p_size=20 --drop=0.0 --attn_drop=0.0 --path_drop=0.0 --optimizer=AdamW --lr_warmup_fn=linear --arch=ViT --augmentation=2 --exp_name=t4-arch"



### s1 Swin Transformer
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --embed_dim=96 --p_size=4 --drop=0.0 --attn_drop=0.0 --path_drop=0.1 --optimizer=AdamW --lr_warmup_fn=linear --arch=Swin --img_size=128 --augmentation=2 --exp_name=s1-base"
# sbatch train.sh "--epochs=25 --batch_size=256 --lr=1e-4 --embed_dim=96 --p_size=8 --drop=0.0 --attn_drop=0.0 --path_drop=0.1 --optimizer=AdamW --lr_warmup_fn=linear --arch=Swin --img_size=128 --augmentation=2 --exp_name=s1-base"

