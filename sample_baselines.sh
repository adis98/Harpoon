#!/bin/bash

# Define the options for the synth_mask parameter
options_dataset=("adult" "bean" "california" "default" "gesture" "letter" "magic" "news" "shoppers")
#options_dataset=("adult")
options_mask=("MAR" "MCAR" "MNAR")
options_ratio=("0.25" "0.5" "0.75")

for dataset in "${options_dataset[@]}"
do
  for mask in "${options_mask[@]}"
  do
    for ratio in "${options_ratio[@]}"
    do
      python3.12 sampling_diffputer.py --dataname $dataset --mask $mask --ratio $ratio --num_steps 50 --num_trials 5
    done
  done
done
#options_dataset=("AustraliaTourism" "MetroTraffic" "RossmanSales" "PanamaEnergy" "BeijingAirQuality")
#options_stride=(8)
# Loop through all synth_mask options and run the Python script with each one
#for dataset in "${options_dataset[@]}"
#do
#  for encoding in "${options_encoding[@]}"
#  do
#    for synth_mask in "${options_synth_mask[@]}"
#    do
#      for s in "${options_stride[@]}"
#      do
#        if [[ "$encoding" == "std" ]]; then
#          python3.12 synthesis_wavestitch_pipeline_strided_preconditioning.py -d $dataset -synth_mask $synth_mask -stride $s
#        elif [[ "$encoding" == "prop" ]]; then
#          python3.12 synthesis_wavestitch_pipeline_strided_preconditioning.py -d $dataset -synth_mask $synth_mask -stride $s -propCycEnc True
#        elif [[ "$encoding" == "onehot" ]]; then
#          python3.12 synthesis_wavestitch_pipeline_strided_preconditioning_onehot.py -d $dataset -synth_mask $synth_mask -stride $s
#        elif [[ "$encoding" == "ordinal" ]]; then
#          python3.12 synthesis_wavestitch_pipeline_strided_preconditioning_ordinal.py -d $dataset -synth_mask $synth_mask -stride $s
#        fi
#      done
#    done
#  done
#done

