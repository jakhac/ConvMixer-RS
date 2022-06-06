#!/bin/bash

### test
sbatch train.sh "--epochs=15 --batch_size=256 --lr=0.0001 --h=1000 --depth=8 --optimizer=Adam --exp_name=v4-addmetrics"
# sbatch train.sh "--epochs=2 --batch_size=2048 --lr=0.001 --h=5 --depth=2 --optimizer=Adam --exp_name=testf1ap"

### baseline
## reasonably large baseline: epochs way to high, early overfit -> lower lr
# sbatch train.sh "--epochs=100 --batch_size=128 --lr=0.001 --h=1024 --depth=8 --optimizer=Adam --exp_name=v0-baseline"
# sbatch train.sh "--epochs=100 --batch_size=128 --lr=0.001 --h=512 --depth=16 --optimizer=Adam --exp_name=v0-baseline"

### decrease learning rate: early overfit -> reduce epochs +-25
# sbatch train.sh "--epochs=40 --batch_size=128 --lr=0.00005 --h=1024 --depth=8 --optimizer=Adam --exp_name=v1-small_lr"
# sbatch train.sh "--epochs=40 --batch_size=128 --lr=0.00005 --h=512 --depth=16 --optimizer=Adam --exp_name=v1-small_lr"

### speed comparison (small model + large batch)
# sbatch train.sh "--epochs=20 --batch_size=512 --lr=0.001 --h=512 --depth=6 --optimizer=Adam --exp_name=v2-speed_comp"

# lower epochs, increase batchsize, regularization?


