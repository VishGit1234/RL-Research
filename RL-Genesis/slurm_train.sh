#!/bin/bash
#SBATCH --job-name=rl_research
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --gres tmpdisk:30480,shard:20480
#SBATCH --time=23:00:00
#SBATCH --output=logs/%j-%x.out  # %j: job ID, %x: job name. Reference: https://slurm.schedmd.com/sbatch.html#lbAH
 
slurm-start-dockerd.sh
export DOCKER_HOST=unix:///tmp/run/docker.sock
docker compose up
