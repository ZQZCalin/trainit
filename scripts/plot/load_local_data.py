# Usage:
#   cd /projectnb/aclab/qinziz/trainit/scripts/plot
#   module load python3/3.10.12 cuda/12.2
#   source /projectnb/aclab/qinziz/trainit/env/bin/activate
#   python load_local_data.py


import json
import wandb
import os


def main():
    api = wandb.Api()
    max_len = 10000
    runs = {
        "baseline_step10k_trapezoid_lr3.33e-4": ("optimizedlearning/test3/c3443384-736b-47e7-bcec-d0d799453c0c",),
        "semi-local_step10k_eps0.12": ("optimizedlearning/test3/wbn8u6q8", "optimizedlearning/test3/9751001"),
        "semi-local_step10k_eps0.24": ("optimizedlearning/test3/0v0qchdy",),
    }
    for name, path in runs.items():
        print(f"Preparing data from {path}...")
        keys = ["iterations", "loss", "lr/schedule"]
        data = { key: [] for key in keys }
        it = 0
        for p in path:
            run = api.run(p)
            rows = run.scan_history(keys=keys[1:])
            for r in rows:
                it += 1
                data["iterations"].append(it)
                data["loss"].append(r.get("loss"))
                data["lr/schedule"].append(r.get("lr/schedule"))
        for k, v in data.items():
            data[k] = v[:max_len]
        os.makedirs(f"local_data/{name}/checkpoint/", exist_ok=True)
        with open(f"local_data/{name}/checkpoint/data.json", "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()