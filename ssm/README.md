# State-Space Models <a target="_blank" href="https://colab.research.google.com/drive/19cpCNgpnvuZ115h--YjW3AzpIR_zUnNe?usp=sharing"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

This task explores efficient long-sequence modeling using the Selective State-Space (S4) model, which reduces the quadratic complexity of Transformers to $\mathcal{O}(L\log L)$ while retaining strong expressivity. A simplified S4 is implemented in PyTorch and evaluated on the sequential CIFAR-10 dataset. Different discretization schemes (Bilinear, Async, Dirac, ZOH) are also discussed.

## Code
The [s4.ipynb](https://github.com/souhhmm/SAiDL-Spring-Assignment-2025/blob/main/ssm/s4.ipynb) notebook has the implementation consisting of:
- [s4a.py](https://github.com/souhhmm/SAiDL-Spring-Assignment-2025/blob/main/ssm/s4a.py) - Async Discretization
- [s4b.py](https://github.com/souhhmm/SAiDL-Spring-Assignment-2025/blob/main/ssm/s4b.py) - Bilinear Discretization
- [s4d.py](https://github.com/souhhmm/SAiDL-Spring-Assignment-2025/blob/main/ssm/s4c.py) - Dirac Discretization
- [s4z.py](https://github.com/souhhmm/SAiDL-Spring-Assignment-2025/blob/main/ssm/s4z.py) - ZOH Discretization
- [main.py](https://github.com/souhhmm/SAiDL-Spring-Assignment-2025/blob/main/ssm/main.py) - Training and Testing

The models were trained on Kaggle (P100). To run the code, simply execute the notebook after choosing the desired discretization scheme.

## Report
The [report](https://github.com/souhhmm/SAiDL-Spring-Assignment-2025/blob/main/ssm/ssm.pdf) has three main sections: Theoretical Understanding, Implementation Details and Results. The theoretical section represents my effort to grasp the underlying concept. The implementation details highlight the code aspect. The results contain the test accuracies of S4 models with different discretization schemes.

The Wandb report can be found [here](https://api.wandb.ai/links/souhhmm-bits-pilani/l1qrt10j). To log using Wandb enter your [key](https://wandb.ai/authorize) in `wandb.login()`.

## References
1. The (one and only) [Annotated S4](https://srush.github.io/annotated-s4/) blog.
2. Albert Gu's [talk](https://www.youtube.com/watch?v=luCBXCErkCs&t=1029s&ab_channel=StanfordMedAI) on Stanford MedAI's channel.
3. After some digging (and stalking?) found this amazing [talk](https://www.youtube.com/watch?v=cAfrHICmDk0&ab_channel=SocietyforAIandDeepLearning) on SAiDL's channel 🫡.
4. Umar Jamil's [talk](https://www.youtube.com/watch?v=8Q_tqwpTpVU&t=4022s&ab_channel=UmarJamil).

Other references are mentioned in the report.