#!/bin/bash
#SBATCH --job-name=rl_research
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --gres tmpdisk:30480,shard:10240
#SBATCH --time=10:00:00
#SBATCH --output=logs/%j-%x.out  # %j: job ID, %x: job name. Reference: https://slurm.schedmd.com/sbatch.html#lbAH
 
slurm-start-dockerd.sh
export DOCKER_HOST=unix:///tmp/run/docker.sock
docker compose up
