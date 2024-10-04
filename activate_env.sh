#!/bin/bash -l

# cd /projectnb/aclab/qinziz/trainit
module load python3/3.10.12 cuda/12.2
source env/bin/activate
python check_env.py