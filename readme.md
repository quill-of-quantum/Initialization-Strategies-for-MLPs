# DeepMLP Initialization & Activation Function Experiments

## 📌 Project Overview

This project aims to systematically compare the effects of different **activation functions** and **weight initialization strategies** on the training performance of deep multilayer perceptrons (MLPs).  
We conduct our experiments using **PyTorch**, evaluating a wide range of network depths (e.g., 5 to 50 layers).

---

## 🚀 Experiment Setup & Recommendations

We initially implemented the framework using **NumPy** integrated with the `courselib` educational framework.  
However, this version was found to be significantly slow (≈ 20 seconds per epoch on average).  
To efficiently evaluate a wide variety of activation and initialization combinations, we transitioned to a **PyTorch GPU-accelerated version** (also can run without GPU on Mac), reducing the average epoch time to about 3 seconds.

- ✅ The main experiment is implemented in [`run.ipynb`](./run.ipynb) using PyTorch with GPU acceleration.
- 🧪 The original NumPy + `courselib` implementation is retained at the end of the notebook for verification and comparison.
- 📄 Dependencies are listed in [`requirements.txt`](./requirements.txt).

> ⚠️ **Running the full set of experiments (4 activations × 8 initializations × multiple layer configurations) takes approximately 8 hours.**

---

## 📁 Project Structure

```bash
.
├── courselib/            # Educational utilities provided by the instructor
├── data/                 # Auto-downloaded MNIST dataset
├── he_factor_results/    # Results from Section 10: He Initialization Parameter (Factor) Comparison
├── results/              # Main experiment outputs (from PyTorch implementation)
├── results_numpy/        # Placeholder for NumPy experiment results (currently empty)
├── sigmoid/              # Additional appendix experiments using Sigmoid
├── run.ipynb             # Main experiment notebook (PyTorch + GPU)
└── requirements.txt      # Python dependencies
```

### ✅ Suggested Usage for Quick Testing

If you only wish to run a subset of the experiments or quickly validate the setup:

- Use the precomputed experiment results and visualization figures provided in the repository;
- Or modify the configuration parameters near the top of `run.ipynb`, for example:
```python
CONFIG = {
    "layers_list": [5, 30, 50],
    "init_types": [
        "he_normal", #"he_uniform",
        "xavier_normal", #"xavier_uniform",
        "normal", #"zero",
        "orthogonal", #"trunc_normal"
    ],
    "activations": ["relu", "tanh"], #, "sigmoid", "gelu"
    "init_params": {
        "he_normal":   {"factor": 2.0, "mode": "fan_in", "nonlinearity": "relu"},
        "he_uniform":  {"factor": 2.0, "mode": "fan_in", "nonlinearity": "relu"},
        "orthogonal":  {"gain": 1.0},
        "trunc_normal": {"mean": 0.0, "std": 0.1, "a": -2.0, "b": 2.0}
    },
    "hidden_size": 128,
    "input_size": 784,
    "output_size": 10,
    "num_epochs": 5,
    "lr": 0.01,
    "batch_size": 128,
    "save_dir": "results"
}
```

## 🗂️ Notebook Contents Overview

The notebook is structured into the following sections:

1. **Configuration Parameters**  
   Definition of hyperparameters, network depths, activation functions, initialization strategies, and other global config options.

2. **Environment Setup**  
   Reproducibility settings such as fixed random seeds and deterministic backend flags.

3. **Model Definition & Training Logic**  
   MLP model construction, activation function injection, weight initialization strategies, and implementation of training/evaluation loops with hooks.

4. **Main Routine**  
   Full automated experiment execution over all combinations of activation functions, initializations, and layer counts, with CSV result export.

5. **Compare Different Initialization Strategies (Fixed Activation)**  
   Visualization of training accuracy and convergence speed when using different initialization strategies with the same activation function.

6. **Compare Different Activation Functions (Fixed Initialization)**  
   Analysis of performance variation across different activation functions under the same weight initialization.

7. **Activation Mean and Std Visualization**  
   Layer-wise tracking of activation statistics (mean and std) across training epochs.

8. **Gradient Norm Visualization**  
   Layer-wise visualization of gradient L2 norms at selected epochs, useful for diagnosing gradient vanishing/explosion.

9. **Accuracy Generalization Comparison Across Different Layer Depths**  
   Final training and test accuracy comparison across networks of varying depths and different initialization strategies.

10. **He Initialization Parameter (Factor) Comparison**  
    Compares the effect of different `factor` values in He initialization on training performance, using fixed activation and architecture.
    
--- 

### ⏱️ Runtime and Logging Details

Each model records **epoch-level training duration** for timing analysis.  
In addition, during the full experiment sweep, the main routine tracks the **overall runtime** and estimates the **expected remaining time (ETA)** after each model finishes training.  
This allows real-time monitoring of experiment progress across all activation–initialization–depth combinations.

For each model run, the following metrics are logged:

- **train_loss** – Training loss per epoch  
- **test_loss** – Validation loss per epoch  
- **train_acc** – Training accuracy per epoch  
- **test_acc** – Validation accuracy per epoch  
- **epoch_time** – Time taken per epoch (in seconds)  
- **act_mean_i** – Mean of activations at layer `i` (per epoch)  
- **act_std_i** – Standard deviation of activations at layer `i`  
- **grad_norm_i** – L2 norm of gradients at layer `i`  

All recorded metrics are exported as `.csv` files to the `results/` directory for downstream analysis and plotting.

## 📎 Appendix: NumPy Implementation (Using `courselib`)

A NumPy-based implementation using the `courselib` educational framework is included at the end of the notebook.  
It is intended for reference and correctness verification, but not used in main experiments due to slower performance.

## 📎 Appendix: Extended Experiment on Sigmoid Activation

During early testing, we found that using **sigmoid activation** with a **learning rate** of `0.01` over `50` epochs was insufficient for proper convergence.  
As a result, we conducted a **dedicated set of experiments** focusing solely on sigmoid, adjusting parameters to allow deeper training and meaningful comparison.

This extended experiment explores how different initialization strategies affect training performance **specifically under sigmoid activation**.

To save storage space and reduce repository size:

- ✅ We include the **source code and result plots** for this sigmoid-specific experiment.
- 🚫 We do **not include the raw training logs or CSV data** for this part.

You can find the results and source in the [`sigmoid/`](./sigmoid/) subdirectory.