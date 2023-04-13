#!/bin/bash

#----- Specify your e-mail address below:
#SBATCH --mail-user=scs76@cam.ac.uk

#----- Specify your job name (without space). This appears in the notificaiton mails:
#SBATCH --job-name=MIN_ML

#----- Specify minimum nodes required (optional):
#SBATCH --nodes=1

#----- Specify minimum RAM/Memory required per CPU (optional):
#SBATCH --mem-per-cpu=8G

#----- Specify minimum CPU cores required (optional):
#SBATCH --cpus-per-task=40

#------------------------------------------------------------------------#
#SBATCH --mail-type=START,FAIL,END
#SBATCH --output=%u-%j.log   
#------------------------------------------------------------------------#

#----- Specify your command(s)/script(s)/job(s)/task(s) below:
module load anaconda3

my_parallel="parallel $SLURM_NTASKS"
my_srun="srun --export=all --exclusive -n1 --cpus-per-task=1 "
$my_parallel "$my_srun python Train_Supervised_cluster.py" ::: {1..2}

# ----- END OF FILE ----- #
