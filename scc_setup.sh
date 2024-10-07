#!/bin/bash -l

module load python3/3.10.12 cuda/12.2

[ ! -d "env" ] && python -m venv env

source env/bin/activate
pip install -r requirements.txt

# manually download jax to match cuda version; Updated 10/07: fix to stable version v0.4.31
pip install --upgrade "jax[cuda12]==0.4.31"
