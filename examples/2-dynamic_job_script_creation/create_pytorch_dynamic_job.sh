#!/bin/bash

echo "Starting PyTorch training experiments..."

# Define the path to your CSV file
CSV_FILE="pytorch_experiments_args.csv"
echo "Reading arguments from CSV file: $CSV_FILE"

# Skip the header line and read the CSV line by line
tail -n +2 "$CSV_FILE" | while IFS=',' read -r source_train target_train source_dev target_dev source_test target_test walltime nodes ppn gpus mem
do
    echo "Processing line: $source_train, $target_train, $source_dev, $target_dev, $source_test, $target_test, $walltime, $nodes, $ppn, $gpus, $mem"

    # Count the number of lines in the trainset source file to get the corpus size
    corpus_size=$(wc -l < "${source_train}" | tr -d ' ')
    echo "Corpus size for $source_train: $corpus_size lines"

    # make a timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    echo "Timestamp for job: $timestamp"

    # Construct the jobname using source and target language codes, corpus size, and timestamp
    jobname="PyTorch-${corpus_size}lines-${timestamp}"
    echo "Job name: $jobname"

    # Construct the command with the extracted parameters
    pytorch_train_command="python src/gpu_pytorch_script_with_args.py --source_train '${source_train}' --target_train '${target_train}' --source_dev '${source_dev}' --target_dev '${target_dev}' --source_test '${source_test}' --target_test '${target_test}'"
    echo "PyTorch command: $pytorch_train_command"

    # Construct the PBS script command with the job arguments
    job_script="qsub -l nodes=${nodes}:ppn=${ppn}:gpus=${gpus} -l mem=${mem} -l walltime=${walltime} -N ${jobname} -v COMMAND='${pytorch_train_command}' train_pytorch_job.pbs -A starting_2024_034"
    echo "PBS job script command: $job_script"

    # Execute the PBS script command
    eval "$job_script"

    # Sleep for 2 seconds to avoid submitting jobs too quickly
    sleep 2
done

echo "Finished submitting all PyTorch experiments."
