# Numerical simulation in "The third pillar of causal analysis? A measurement perspective on causal representations"

This directory contains the code to reproduce the numerical simulations in the paper "The third pillar of causal analysis? A measurement perspective on causal representations" [1].

The Test-based Measurement EXclusivity (T-MEX) score proposed in [1] uses conditional independence (CI) tests to evaluate a Causal Representation Learning (CRL) algorithm. In particular, it evaluates the exclusivity of information captured by learned representations about the underlying causal factors. Example implementations of T-MEX score using different CI tests can be found in `eval_numerical.py`, including the Covariance Measure Tests implemented in [pycoments](https://github.com/shimenghuang/pycomets) and the Kernel-based Conditional Independence Test (KCI) implemented in [causal-learn](https://causal-learn.readthedocs.io/en/latest/independence_tests_index/kci.html).

The main numerical simulation (Section 5.1 in [1]) uses the multi-view with partial observability by [2] whose dependencies can be installed as below. 

## Instal dependencies of Multi-View CRL with Partial Observability

```shell
cd $PROJECT_DIR
mamba env create -f env.yaml
mamba activate crl_venv
pre-commit install
```

## Reproduce Numerical Simulations (Section 5.1) in [1]
```shell
# train
python main_numerical.py --model-id "five_latents_a" --n-steps 50001 --seed 858
python main_numerical.py --model-id "five_latents_b" --n-steps 51 --seed 134 

# evaluate
python main_numerical.py --model-id "five_latents_a" --evaluate --seed 426
python main_numerical.py --model-id "five_latents_b" --evaluate --seed 680
python eval_numerical.py # compute T-MEX scores and other metrics
```

## References
[1] D. Yao, S. Huang, R. Cadei, K. Zhang, & F. Locatello. "The Third Pillar of Causal Analysis? A Measurement Perspective on Causal Representations". arXiv preprint [arXiv:2505.17708](https://arxiv.org/abs/2505.17708). Accepted at NeurIPS 2025. 

[2] D. Yao, D. Xu, S. Lachapelle, S. Magliacane, P. Taslakian, G. Martius, J. von Kügelgen, and F. Locatello. "Multi-view causal representation learning with partial observability." In Proc. 12th Int. Conf. Learn. Representations (ICLR), Vienna, Austria, 2024
