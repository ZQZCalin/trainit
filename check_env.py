# Check Python library versions.
divider = "="*100
print(divider+"\nchecking libraries...")
import importlib

# List of library names to check versions for
libraries = [
    "jax",
    "numpy",
    "scipy",
    "pandas",
    "torch",
    "transformers",
    "optax",
    "equinox",
    "datasets",
    # Add any other libraries you're interested in
]

for lib in libraries:
    try:
        # Dynamically import the module
        module = importlib.import_module(lib)
        # Attempt to print the version
        print(f"{lib}: {module.__version__}")
    except ImportError as err:
        print(f"{lib}: Not installed\n  {err}")
    except AttributeError as err:
        print(f"{lib}: Version not accessible via __version__\n  {err}")


# Check if CUDA is available.
print(divider+"\nchecking cuda state...")
import torch
print("CUDA Available:", torch.cuda.is_available())
# List all available GPUs
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("GPU:", torch.cuda.get_device_name(i))
else:
    print("No GPUs found")


# Check JAX device usage.
print(divider+"\nchecking jax devices...")
import jax
print("Devices:", jax.devices())