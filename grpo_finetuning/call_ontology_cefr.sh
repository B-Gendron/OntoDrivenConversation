#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <batch_index> <file_path> <experiment> <repo>"
  exit 1
fi

# Assign the arguments to variables
batch_index="$1"
file_path="$2"
experiment="$3"
repository="$4"

# Move to the other GitHub repo
initial_path="../grpo_finetuning/code"
new_path="../../onto_processing/"
cd $new_path
# echo Change location to $new_path
script_path="code/finetuning_batch_individuals_cefr.py"

# Check if the script file exists
if [ ! -f "$script_path" ]; then
  echo "Error: Python script not found at $script_path"
  exit 1
fi

# Run the Python script with the provided arguments
python3 "$script_path" -i "$batch_index" -p "$file_path" -e "$experiment" -r "$repository"
# echo Successfully saved ontology individual!

cd $initial_path
# echo Change location back to $initial_path
