# HPC LT3 Tutorial

This repository provides examples and scripts to help you get started with using High-Performance Computing (HPC) for NLP research. The examples cover basic job submissions, using GPUs, and dynamically creating job scripts.

```sh
git clone https://github.com/TomMoeras/hpc_lt3_tutorial
cd hpc_lt3_tutorial
```

## Directory Structure

```bash
hpc_lt3_tutorial/
├── README.md
├── examples/
│   ├── 0-basic_jobs/
│   │   ├── A-simple_shell_job/
│   │   │   ├── A-simple_shell_job.pbs
│   │   ├── B-simple_python_job/
│   │   │   ├── B-simple_python_job.pbs
│   │   │   ├── B-simple_python_script.py
│   ├── 1-using_gpu/
│   │   ├── gpu_pytorch_job.pbs
│   │   ├── gpu_pytorch_script.py
│   ├── 2-dynamic_job_script_creation/
│   │   ├── create_pytorch_data.sh
│   │   ├── create_pytorch_dynamic_job.sh
│   │   ├── gpu_pytorch_script.py
│   │   ├── pytorch_experiments_args.csv
│   │   ├── train_pytorch_job.pbs
│   │   └── data/
│   ├── 3-finetuning_llm/
│       ├── environment.yml
│       ├── finetune_llm_job.pbs
│       ├── requirements.txt
│       ├── data/
│       ├── src/
```

## Getting Started

### 0. Basic Jobs

#### A. Simple Shell Job

Navigate to the simple shell job directory and submit the job script.

```sh
cd examples/0-basic_jobs/A-simple_shell_job
qsub simple_shell_job.pbs
```

This script will run a basic shell command like printing a message and calculate Fibonacci numbers.

#### B. Simple Python Job

Navigate to the simple Python job directory and submit the job script.

```sh
cd examples/0-basic_jobs/B-simple_python_job
qsub simple_python_job.pbs
```

This script will load Python and execute a simple Python script without external libraries.

### 1. Using GPU

Navigate to the GPU job directory and submit the job script.

```sh
cd examples/1-using_gpu
ml swap cluster/accelgor
qsub gpu_pytorch_job.pbs
```

This script will load a PyTorch model and perform a simple NLP task using GPU resources.

### 2. Dynamic Job Script Creation

This example demonstrates how to dynamically create and submit job scripts based on a CSV file with experiment parameters.

```sh
cd examples/2-dynamic_job_script_creation
ml swap cluster/accelgor
source create_pytorch_dynamic_job.sh
```

This will read arguments from `pytorch_experiments_args.csv` and submit multiple jobs accordingly.

### 3. Fine-tuning a Language Model

Navigate to the fine-tuning LLM job directory and submit the job script.

```sh
cd examples/3-finetuning_llm
ml swap cluster/accelgor
qsub finetune_llm_job.pbs
```

This script will set up the environment and fine-tune a large language model using the provided data and configurations.

## Notes

- Make sure to run all scripts from their respective directories to ensure correct file paths and execution contexts.
- Adjust the resource requests (e.g., nodes, ppn, gpus, mem, walltime) in the job scripts as necessary based on your specific requirements and the availability of resources on the HPC cluster.
