# Core ML <a target="_blank" href="https://colab.research.google.com/drive/1PUj50F3kqv3JKEtW7W-IJ8d1RwNCLYtK?usp=sharing"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

This task focuses on robustness in machine learning under noisy labels using normalized loss functions and the _Active Passive Loss (APL)_ framework. Symmetric and asymmetric noise are introduced to CIFAR-10, and models are trained with NCE, NFL, MAE, and RCE.

## Code
The [core.ipynb](https://github.com/souhhmm/SAiDL-Spring-Assignment-2025/blob/main/core/core.ipynb) notebook implements and evaluates different loss functions. The experiments have been run on Colab (T4) and Kaggle (P100). To run the code, simply execute the notebook after uncommenting the desired loss function in the main section with the desired config.

## Report
The [report](https://github.com/souhhmm/SAiDL-Spring-Assignment-2025/blob/main/core/core.pdf) is structured into two main sections: Theoretical Understanding and Empirical Study. The theoretical section represents my effort to grasp the underlying concepts, while the empirical study focuses on the experiments conducted to evaluate different loss functions under noisy labels.

The Wandb report can be found [here](https://api.wandb.ai/links/souhhmm-bits-pilani/31ysfcz1). To log using Wandb enter your [key](https://wandb.ai/authorize) in `wandb.login()`. 

## References
The official [paper](https://arxiv.org/abs/2006.13554) and [code](https://github.com/HanxunH/Active-Passive-Losses/tree/master).