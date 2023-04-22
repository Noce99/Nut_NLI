#!/bin/sh
#SBATCH --partition=GPUQ # CHANGE THIS TO CPUQ IF YOU DON'T NEED GPUs!!
#SBATCH --account=ie-idi # READ THIS TO FIND YOUR ACCOUNT: https://www.hpc.ntnu.no/idun/getting-started-on-idun/accounting/
#SBATCH --mem=64GB # Total memory you need. Most DL jobs should do fine with 64GB.
#SBATCH --nodes=1 # Number of nodes. Leave this to 1 as long as you're not running multi-node/multi-gpu training
#SBATCH --cpus-per-task=4 # Number of CPU cores required. Most V100 nodes has 4 CPU cores per GPU, so to maximize usage of the nodes, leave it to 4 (unless you know that your specific node has more than 4 cores/GPU).
#SBATCH --job-name=anli # Set this to whatever you want
#SBATCH --output=/cluster/home/enricoma/out_3.slurm # The output file. NOTE! The output directory (/cluster/home/haakohu/ in this case)  has to exist before you submit the job!
#SBATCH --time=00:10:00 # Maximum running time of your job.
#SBATCH --export=ALL # Export all environment variables.
#SBATCH --gres=gpu:V100:1 # GPU type and number of GPUs.
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Total of GPUS: $CUDA_VISIBLE_DEVICES"
nvidia-smi
nvidia-smi nvlink -s
nvidia-smi topo -m
module purge
module load cuDNN/8.4.1.50-CUDA-11.7.0
cd /cluster/home/enricoma/NutNLI/Nut_NLI
~/python nut_training.py
