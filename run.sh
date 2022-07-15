#!/bin/bash
#SBATCH -A ict22_esp_0
#SBATCH -p m100_usr_prod
#SBATCH --time 24:00:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=0
# --ntasks-per-node=8   # 8 tasks out of 128
#SBATCH --gres=gpu:1          # 1 gpus per node out of 4
#SBATCH --job-name=preprocessing
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=vblasone@ictp.it
#SBATCH -o /m100_work/ICT22_ESP_0/vblasone/rainfall-maps/.log/preprocessing_input.out
#SBATCH -e /m100_work/ICT22_ESP_0/vblasone/rainfall-maps/.log/preprocessing_input.err

module load --auto python hdf5

cd /m100_work/ICT22_ESP_0/vblasone/rainfall-maps

python3 preprocessing.py
