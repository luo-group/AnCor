#!/bin/bash

dataset=OBSCN_HUMAN_Tsuboyama_2023_1V1C


accelerate launch --config_file config/parallel_config.yaml ancor/train.py \
--config config/training_config.yaml \
--dataset $dataset \
--sample_seed 0 \
--model_seed 1 \
--shot 72 \
--prefix ancor_72shot 
	