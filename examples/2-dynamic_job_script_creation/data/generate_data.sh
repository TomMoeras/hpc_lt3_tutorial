#!/bin/bash

# Define base directory
BASE_DIR="examples/2-dynamic_job_script_creation/data"

# Define directories for run1 and run2
RUN1_DIR="${BASE_DIR}/run1"
RUN2_DIR="${BASE_DIR}/run2"

# Create directories
mkdir -p "$RUN1_DIR"
mkdir -p "$RUN2_DIR"

# Function to create data files
create_data_files() {
    local dir=$1
    local source_train=$2
    local target_train=$3
    local source_dev=$4
    local target_dev=$5
    local source_test=$6
    local target_test=$7

    echo -e "$source_train" > "${dir}/source_train.txt"
    echo -e "$target_train" > "${dir}/target_train.txt"
    echo -e "$source_dev" > "${dir}/source_dev.txt"
    echo -e "$target_dev" > "${dir}/target_dev.txt"
    echo -e "$source_test" > "${dir}/source_test.txt"
    echo -e "$target_test" > "${dir}/target_test.txt"
}

# Data for run1
create_data_files "$RUN1_DIR" \
"Hello HPC with GPU\nThis is a great example\nWe are running experiments" \
"positive\npositive\npositive" \
"This is a test sentence\nExperimenting with HPC" \
"negative\npositive" \
"Running tests on HPC\nAnother test sentence" \
"positive\nnegative"

# Data for run2
create_data_files "$RUN2_DIR" \
"NLP tasks are fun\nLearning PyTorch on HPC\nTraining with data" \
"positive\npositive\npositive" \
"Validating the model\nTesting PyTorch script" \
"positive\nnegative" \
"Inference with model\nFinal test sentence" \
"positive\nnegative"

echo "Data files created successfully."
