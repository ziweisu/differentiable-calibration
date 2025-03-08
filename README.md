# Differentiable Calibration of Inexact Stochastic Simulation Models

This repository contains the implementation code and raw results for the paper "Differentiable Calibration of Inexact Stochastic Simulation Models via Kernel Score Minimization" published at AISTATS 2025.

## Overview

We propose a novel frequentist method to learn and quantify the uncertainties of differentiable input parameters of inexact stochastic simulation models using output-level data via kernel score minimization with stochastic gradient descent. Our approach:

- Works effectively when only output-level data is available
- Addresses model inexactness (discrepancies between the model and target system)
- Provides valid confidence sets for input simulation parameters
- Outperforms both Bayesian and frequentist baselines in extensive experiments

# Repository Structure

 * [src](./src)
   * [estimation.py](./src/estimation.py) - Parameter and CI estimation for 1D case
   * [estimation_2d.py](./src/estimation_2d.py) - Parameter and CI estimation for 2D case
   * [kernels.py](./src/kernels.py) - Implementation of kernel functions (Riesz, Gaussian)
   * [plot.py](./src/plot.py) - Plotting utilities
   * [scoring_rules.py](./src/scoring_rules.py) - Implementation of scoring rules
   * [simulation_model.py](./src/simulation_model.py) - Simulation of G/G/1 queueing model (1D)
   * [simulation_model_2d.py](./src/simulation_model_2d.py) - Simulation G/G/1 queueing model (2D)
 * [experiments](./experiments)
   * [1d_exact.ipynb](./experiments/1d_exact.ipynb) - Experiment 1: Exact G/G/1 with Riesz Kernel (Table 4)
   * [1d_exact_Gaussian.py](./experiments/1d_exact_Gaussian.py) - Experiment 1: Exact G/G/1 with Gaussian Kernel (Table 4)
   * [1d_inexact.py](./experiments/1d_inexact.py) - Experiment 2: Inexact G/G/1 with Riesz Kernel (Table 2)
   * [1d_inexact_Gaussian.py](./experiments/1d_inexact_Gaussian.py) - Experiment 2: Inexact G/G/1 with Gaussian Kernel (Table 2)
   * [2d_exact_nonstat.py](./experiments/2d_exact_nonstat.py) - Experiment 3: Non-stationary 2D exact G/G/1 with Riesz Kernel (Table 5)
   * [2d_exact_nonstat_Gaussian.py](./experiments/2d_exact_nonstat_Gaussian.py) - Experiment 3: Non-stationary 2D exact G/G/1 with Gaussian Kernel (Table 5)
   * [2d_inexact_stat.py](./experiments/2d_inexact_stat.py) - Experiment 4: Stationary 2D inexact G/G/1 with Riesz Kernel (Table 3)
   * [2d_inexact_stat_Gaussian.py](./experiments/2d_inexact_stat_Gaussian.py) - Experiment 4: Stationary 2D inexact G/G/1 with Gaussian Kernel (Table 3)
   * [3d_stoc_volatility.py](./experiments/3d_stoc_volatility.py) - 3D Stochastic volatility model (Table 13 & 14) with implementation of simulation, SGD and CI estimation included
   * ... followed by other experiment variants, ES (Eligbility Set) and rejectionABC baselines
 * [results](./results)
   * [dataframes](./results/dataframes) - Tabular results from experiments, folders follow the same naming convention as in [experiments](./experiments) except:
     *  [stochastic_volatility](./results/dataframes/stochastic_volatility) - Tabular results from Table 13, Riesz Kernel
     *  [stochastic_volatility_Gaussian](./results/dataframes/stochastic_volatility_Gaussian) - Tabular results from Table 13, Gaussian Kernel
     *  [stochastic_volatility_corrected](./results/dataframes/stochastic_volatility_corrected) - Tabular results from Table 14, Riesz Kernel
     *  [stochastic_volatility_Gaussian_corrected](./results/dataframes/stochastic_volatility_Gaussian_corrected) - Tabular results from Table 14, Gaussian Kernel
   * [plots](./results/plots) - Generated figures and visualizations, folders follow the same naming convention as in [dataframes](./results/dataframes)
 * [.gitignore](./.gitignore)
 * [README.md](./README.md)
 * [requirements.txt](./requirements.txt)
 * [LICENSE](./LICENSE)

For the NPL-MMD baseline, please go to https://github.com/ziweisu/npl_mmd_project for the implementation and results.
## Requirements

- Python >= 3.8
- PyTorch with CUDA support (if CUDA is available):
  For PyTorch with CUDA support, you may need to follow the instructions at https://pytorch.org/get-started/locally/ to install the appropriate version that matches your CUDA version.
- Dependencies (for local CPU)
```console
numpy>=1.19.0
pandas>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
torch>=1.9.0
tqdm>=4.50.0
scipy>=1.6.0
```
- Install dependencies
```console
pip install -r requirements.txt
```

## Running Experiments
```console
# Experiment 1: Exact G/G/1 Queue (Table 4)
python experiments/1d_exact_Gaussian.py
# followed by variants

# Experiment 2: Inexact G/G/1 Queue (Table 2) 
python experiments/1d_inexact.py
# followed by variants

# Experiment 3: Non-stationary exact 2D G/G/1 Queue (Table 5)
python experiments/2d_exact_nonstat.py
# followed by variants

# Experiment 4: Stationary inexact 2D G/G/1 Queue (Table 3)
python experiments/2d_inexact_stat.py
# followed by variants

# Additional experiment: Stochastic Volatility Model
python experiments/3d_stoc_volatility.py
```

Results will be saved to the [results](./results) directory:

- Tables and numerical data in [results/dataframes/](./results/dataframes) 
- Generated figures and visualizations in  [results/plots/](./results/plots) 

## Method Details

Our approach, termed Kernel Optimum Score Estimation (KOSE), uses kernel score minimization via stochastic gradient descent to estimate simulation parameters. Key components:

- Kernel Scoring Rules: We utilize kernel-based scoring rules (Riesz/Energy and Gaussian) that induce the Maximum Mean Discrepancy (MMD).
- Unbiased Gradient Estimation: For stochastic gradient descent, we leverage U-statistic approximations for unbiased gradient estimates.
- Asymptotic Normality: We establish the first asymptotic normality result for kernel optimum score estimators under model inexactness.
- Confidence Set Construction: Based on these asymptotic results, we construct valid confidence sets for simulation parameters.

## Citation

If you use this code in our research, please cite our paper

```console
@inproceedings{sudifferentiable,
  title={Differentiable Calibration of Inexact Stochastic Simulation Models via Kernel Score Minimization},
  author={Su, Ziwei and Klabjan, Diego},
  booktitle={The 28th International Conference on Artificial Intelligence and Statistics}
}

# or the ArXiv version
@article{su2024differentiable,
  title={Differentiable Calibration of Inexact Stochastic Simulation Models via Kernel Score Minimization},
  author={Su, Ziwei and Klabjan, Diego},
  journal={arXiv preprint arXiv:2411.05315},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about the paper or code, please contact Ziwei Su (ziwei.su@northwestern.edu).
