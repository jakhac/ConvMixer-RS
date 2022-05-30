#!/bin/bash

sbatch train.sh "--epochs=5 --batch_size=64 --lr=0.1 --h=128 --depth=2 --ds_size=200 --run_tests_n=3" "test1"
##sbatch train.sh "--epochs=5 --batch_size=64 --lr=0.1 --h=64 --depth=2 --ds_size=200 --run_tests_n=5" "test2"
##sbatch train.sh "--epochs=5 --batch_size=64 --lr=0.001 --h=64 --depth=2 --ds_size=200 --run_tests_n=3" "test3"
##sbatch train.sh "--epochs=5 --batch_size=32 --lr=0.1 --h=64 --depth=2 --ds_size=200 --run_tests_n=3" "test4"
