#!/bin/bash

# Define the options for the synth_mask parameter
options_dataset=("adult" "bean" "california" "default" "gesture" "letter" "magic" "news" "shoppers")


for dataset in "${options_dataset[@]}"
do
  python3.12 train_hyperimpute.py --dataname $dataset
#  python3.12 train_GReaT.py --dataname $dataset
#  python3.12 train_repaint.py --dataname $dataset
#  python3.12 train_diffputer.py --dataname $dataset
done

