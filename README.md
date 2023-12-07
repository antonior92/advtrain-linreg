# Regularization properties of adversarially-trained linear regression




Companion code to the [paper](https://arxiv.org/abs/2310.10807): 
```
Regularization properties of adversarially-trained linear regression
Antônio H Ribeiro, Dave Zachariah, Francis Bach, Thomas B. Sch\"on
NeurIPS 2023 (spotlight)
```

Paper: [arXiv:2310.10807](https://arxiv.org/abs/2310.10807)

Other info:
[open review](https://openreview.net/forum?id=K8gLHZIgVW) - 
[video](https://recorder-v3.slideslive.com/?share=86229&s=006e4a99-1e12-463e-b7f1-6767feb64b7e) - 
[poster](https://antonior92.github.io/pdfs/posters/2023-Neurips.pdf) - 
[slides](https://antonior92.github.io/pdfs/slides/2023-NeurIPS.pdf) -
[NeurIPS poster page](https://nips.cc/virtual/2023/poster/72028) -
[summary tweet](https://twitter.com/ahortaribeiro/status/1732429927784292772)

<details>
<summary><b>Paper Abstract:</b></summary>
<i>State-of-the-art machine learning models can be vulnerable to very small input
perturbations that are adversarially constructed. Adversarial training is an 
effective approach to defend against it. Formulated as a min-max problem, it
searches for the best solution when the training data were corrupted by the 
worst-case attacks. Linear models are among the simple models where vulnerabilities 
can be observed and are the focus of our study. In this case, adversarial training 
leads to a convex optimization problem which can be formulated as the minimization 
of a finite sum. We provide a comparative analysis between the solution of adversarial 
training in linear regression and other regularization methods. Our main findings are 
that: (A) Adversarial training yields the minimum-norm interpolating solution in the 
overparameterized regime (more parameters than data), as long as the maximum disturbance
radius is smaller than a threshold. And, conversely, the minimum-norm interpolator is 
the solution to adversarial training with a given radius. (B) Adversarial training can 
be equivalent to parameter shrinking methods (ridge regression and Lasso). This happens
in the underparametrized region, for an appropriate choice of adversarial radius and 
zero-mean symmetrically distributed covariates. (C) For $\ell_\infty$-adversarial 
training---as in square-root Lasso---the choice of adversarial radius for optimal 
bounds does not depend on the additive noise variance. We confirm our theoretical 
findings with numerical examples.</i>
</details>





## Description

Given pair of input-output samples $(x_i, y_i), i = 1, \dots, n$, adversarial training in linear regression is 
formulated as  a min-max optimization problem:

$$\min_\beta \frac{1}{n} \sum_{i=1}^n \max_{||\Delta x_i|| \le \delta} (y_i - \beta^\top(x_i+ \Delta x_i))^2$$

The paper analyse the properties of this problem and this repository contain code for reproducing the experiements
in the paper.


## Jupyter notebooks
We provide jupiter notebooks with minimal examples that can be used to quickly reproduce some of the paper main results:

*We try to keep these notebooks as simple as possible, removing some of the plot configurations or
running the experiment only once instead of multiple times.*


| Figure   | Jupyter Notebook | Colab | 
| ----- | ---- | ---- |
| Fig. 1 | [![Fig1](https://img.shields.io/badge/Fig1-Notebook-f37626?logo=jupyter&style=flat)](notebooks/fig1.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonior92/advtrain-linreg/blob/main/notebooks/fig1.ipynb) |
| Fig. 2 |  [![Fig2](https://img.shields.io/badge/Fig2-Notebook-f37626?logo=jupyter&style=flat)](notebooks/fig2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonior92/advtrain-linreg/blob/main/notebooks/fig2.ipynb) |
| Fig. 3 |  [![Fig3](https://img.shields.io/badge/Fig3-Notebook-f37626?logo=jupyter&style=flat)](notebooks/fig3.ipynb)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonior92/advtrain-linreg/blob/main/notebooks/fig3.ipynb) |
| Fig. 4 |   [![Fig4](https://img.shields.io/badge/Fig4-Notebook-f37626?logo=jupyter&style=flat)](notebooks/fig4.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonior92/advtrain-linreg/blob/main/notebooks/fig4.ipynb) |


## Requirements

We use `cvxpy` to solve adversarial training optimization problem. We also use standard scientific python packages
`numpy`, `scipy`, `scikit-learn`, `pandas`, `seaborn` and `matplotlib`. See `requirements.txt`.


## Implementing adversarial training


The script **advtrain.py** implement adversarial training using python. It uses the reformulation provided in Proposition 1 in the paper. 
```python
import cvxpy as cp
import numpy as np


def compute_q(p):
    if p != np.Inf and p > 1:
        q = p / (p - 1)
    elif p == 1:
        q = np.Inf
    else:
        q = 1
    return q


class AdversarialTraining:
    def __init__(self, X, y, p):
        m, n = X.shape
        q = compute_q(p)
        # Formulate problem
        param = cp.Variable(n)
        param_norm = cp.pnorm(param, p=q)
        adv_radius = cp.Parameter(name='adv_radius', nonneg=True)
        abs_error = cp.abs(X @ param - y)
        adv_loss = 1 / m * cp.sum((abs_error + adv_radius * param_norm) ** 2)
        prob = cp.Problem(cp.Minimize(adv_loss))
        self.prob = prob
        self.adv_radius = adv_radius
        self.param = param
        self.warm_start = False

    def __call__(self, adv_radius, **kwargs):
        try:
            self.adv_radius.value = adv_radius
            self.prob.solve(warm_start=self.warm_start, **kwargs)
            v = self.param.value
        except:
            v = np.zeros(self.param.shape)
        return v
```


## Experiments

The folders  `minnorm`, `varying_regularization`, `random_projections`, `varying_regularization`, `comparing_methods`, `magic`, `misc`
contain the scripts for different experiment and for generating the figures in the paper


*These allow to reproduce the figures in the paper exactly.
The jupyter notebooks above might be easier to understand, since they contain simplified code.*

- `minnorm/`: Evaluate minimum norm interpolator. Plot Fig. 2 and S4, S5, S6.
- `regulariazation_paths/`: Regularization paths. Plot Fig. 1, 3 and S1, S2, S3.
- `varying_regularization/`: Varying regularization strength. Plot Fig. 4 and S7, S8, S9.
- `misc/`: other scripts. Plot Fig. 5.
- `random_projections/`: Random projections. Plot Fig. S.10.
- `magic/`: Experiment with Magic dataset.  Plot Fig. 6.
- `Comparing methods (magic)`:  Compare methods with and without cross-validation. Plot Fig. S.11
  
Inside some of this folder, we include:
- `plots/`: containing the pdf plots generated from the experiment.
- `results/`: containing results in csv format.


### Running all experiments
You can use **run.sh** to generate all figures in the paper.

```sh
# Description: Script to run all experiments in the paper

### Download data ###
wget http://mtweb.cs.ucl.ac.uk/mus/www/MAGICdiverse/MAGIC_diverse_FILES/BASIC_GWAS.tar.gz
tar -xvf BASIC_GWAS.tar.gz

### Evaluate minimum norm interpolator ###
# ---- Plot Fig. 2 and S4, S5, S6 ---- #
( cd minnorm && sh run.sh )

###  regularization paths ###
# ---- Plot Fig. 1, 3 and S1, S2, S3 ---- #
( cd regularization_paths && sh run.sh )

### varying regularization strength  ###
# ---- Plot Fig. 4 and S7, S8, S9 ---- #
( cd varying_regularization && sh run.sh )

### Random projections ###
# ---- Plot Fig. S.10---- #
( cd random_projections && sh run.sh )

### Other ###
# ---- Plot Fig. 5 ---- #
( cd misc && sh run.sh )

### MAGIC ###
# ---- Plot Fig. 6 ---- #
( cd magic && sh run.sh )

### Comparing methods (magic) ###
# ---- Plot Fig. S.11---- #
( cd comparing_methods && sh run.sh )
```
