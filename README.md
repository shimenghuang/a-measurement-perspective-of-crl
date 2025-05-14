# Numerical simulation in "The third pillar of causal analysis? A measurement perspective on causal representations"

The simulation employs the multi-view with partial observability by 
[1]. 

## Installation of Multi-View CRL with Partial Observability
<p align="left">
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
    <a href="https://pytorch.org/get-started/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1.0-orange.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <a href="https://mamba.readthedocs.io/en/latest/"><img alt="virtualenv" src="https://img.shields.io/badge/virtualenv-mamba-pink.svg"></a>
</p>

```shell
cd $PROJECT_DIR
mamba env create -f env.yaml
mamba activate crl_venv
pre-commit install
```

## Reproduce the results
```shell
# train
python main_numerical.py --model-id "five_latents_a" --n-steps 50001 --seed 858
python main_numerical.py --model-id "five_latents_b" --n-steps 51 --seed 134 

# evaluate
python main_numerical.py --model-id "five_latents_a" --evaluate --seed 426
python main_numerical.py --model-id "five_latents_b" --evaluate --seed 680
python eval_numerical.py 
```

## References
[1] D. Yao, D. Xu, S. Lachapelle, S. Magliacane, P. Taslakian, G. Martius, J. von Kügelgen, and F. Locatello. "Multi-view causal representation learning with partial observability." In Proc. 12th Int. Conf. Learn. Representations (ICLR), Vienna, Austria, 2024
