#!/bin/bash

sbatch train.sh "--epochs=50 --batch_size=64 --lr=0.01 --h=128 --depth=4 --run_tests_n=5"
sbatch train.sh "--epochs=50 --batch_size=64 --lr=0.1 --h=128 --depth=4 --run_tests_n=5"

sbatch train.sh "--epochs=50 --batch_size=128 --lr=0.01 --h=128 --depth=4 --run_tests_n=5"
sbatch train.sh "--epochs=50 --batch_size=128 --lr=0.1 --h=128 --depth=4 --run_tests_n=5"

sbatch train.sh "--epochs=50 --batch_size=64 --lr=0.01 --h=256 --depth=6 --run_tests_n=5"
sbatch train.sh "--epochs=50 --batch_size=64 --lr=0.1 --h=256 --depth=6 --run_tests_n=5"

sbatch train.sh "--epochs=50 --batch_size=128 --lr=0.01 --h=256 --depth=6 --run_tests_n=5"
sbatch train.sh "--epochs=50 --batch_size=128 --lr=0.1 --h=256 --depth=6 --run_tests_n=5"
