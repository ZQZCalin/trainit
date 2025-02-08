# Muon-variants and Mango optimizer

- Run experiments: `./scripts/experiments/preconditioner/mango.sh`
- Visualize model structure and the mango label function (*on an interactive session*):
    ```python
    from optimizers.muon.muon_test import visualize_label_params
    visualize_label_params()
    ```
- See the summary of gpt2 model and mango label outcome in [here](./gpt2_mango_summary.txt).