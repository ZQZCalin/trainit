"""Summarize the entire experiment
"""

import argparse
import wandb
import json
import os


DATA_FNAME = "data.json"
KEYS = ["loss", "accuracy", "iterations", "lr/schedule"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--desc", type=str, default="")
    args = parser.parse_args()

    fname = os.path.join(args.ckpt, DATA_FNAME)
    with open(fname, 'r') as f:
        data = json.load(f)     # data should be a .json with 4 keys: loss, accuracy, iterations, lr/schedule

    wandb.init(project=args.proj, name=args.name, notes=args.desc)
    for loss, acc, iter, lr in zip(*[data[key] for key in KEYS]):
        wandb.log({
            "loss": loss,
            "accuracy": acc,
            "lr/schedule": lr,
        }, step=iter)
    
    wandb.finish()


if __name__ == "__main__":
    main()