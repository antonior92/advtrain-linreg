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
[NeurIPS poster page](https://nips.cc/virtual/2023/poster/72028)

## Description

Given pair of input-output samples $(x_i, y_i), i = 1, \dots, n$, adversarial training in linear regression is 
formulated as  a min-max optimization problem:

$$\min_\beta \frac{1}{n} \sum_{i=1}^n \max_{||\Delta x_i|| \le \delta} (y_i - \beta^\top(x_i+ \Delta x_i))^2$$

The paper analyse the properties of this problem and this repository contain code for reproducing the experiements
in the paper.



## Colab

We provide google colab to reproduce the main experiments in the paper:

| Fig   | Colab | 
| ----- | ---- |
| Fig 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonior92/advtrain-linreg/blob/main/notebooks/fig1.ipynb) |
| Fig 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonior92/advtrain-linreg/blob/main/notebooks/fig3.ipynb) |


## Generating all figures

You can use **run.py** to generate all figures in the paper
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
( cd experiments/regularization_paths && sh run.sh )

### varying regularization strength  ###
# ---- Plot Fig. 4 and S7, S8, S9 ---- #
( cd experiments/varying_regularization && sh run.sh )

### Random projections ###
# ---- Plot Fig. S.10---- #
( cd experiments/random_projections && sh run.sh )

### MAGIC ###
# ---- Plot Fig. 6 ---- #
( cd experiments/magic && sh run.sh )

### Comparing methods (magic) ###
# ---- Plot Fig. S.11---- #
( cd experiments/comparing_methods && sh run.sh )
```

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