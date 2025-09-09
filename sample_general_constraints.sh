#!/bin/bash

# Define the options for the synth_mask parameter
options_dataset=("adult" "default" "shoppers")
options_constraint=("range" "category" "both")

for dataset in "${options_dataset[@]}"
do
  for constraint in "${options_constraint[@]}"
  do
#    python3.12 sampling_harpoon_ohe_tubular_generalconstraints.py --dataname $dataset --constraint $constraint
     python3.12 sampling_repaint_generalconstraints.py --dataname $dataset --constraint $constraint
  done
done
sendemail -f aditya.ssr@gmail.com -t aditya.ssr@gmail.com -u "Experiment Complete" -s smtp.gmail.com:587 -o tls=yes -xu aditya.ssr@gmail.com -xp zfclbeznrksnrhfs -m "DiffPuter general constraints"

