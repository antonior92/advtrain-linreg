# Regularization properties of adversarially-trained linear regression




Companion code to the [paper](https://arxiv.org/abs/2310.10807). 

Citation
```
Regularization properties of adversarially-trained linear regression
Antônio H Ribeiro, Dave Zachariah, Francis Bach, Thomas B. Sch\"on
NeurIPS 2023 (spotlight)
```

Paper:
- [arXiv:2310.10807](https://arxiv.org/abs/2310.10807)
- [NeurIPS poster](https://nips.cc/virtual/2023/poster/72028)

Other info:
[open review](https://openreview.net/forum?id=K8gLHZIgVW) - 
[video](https://recorder-v3.slideslive.com/?share=86229&s=006e4a99-1e12-463e-b7f1-6767feb64b7e) - 
[poster](pdfs/posters/2023-Neurips.pdf) - 
[slides](pdfs/slides/2023-slides.pdf)

## Description

Given pair of input-output samples $(x_i, y_i), i = 1, \dots, n$, adversarial training in linear regression is 
formulated as  a min-max optimization problem:

$$\min_\beta \frac{1}{n} \sum_{i=1}^n \max_{||\Delta x_i|| \le \delta} (y_i - \beta^\top(x_i+ \Delta x_i))^2$$

The paper analyse the properties of this problem and this repository contain code for reproducing the experiements
in the paper.


## Folder content
- The file `advtrain.py` to a solution of adverasarial training problem using CvxPy (and is used by the other scripts).
- The scripts`fig1and3_regularization_path.py`: generate Figures 1 and 3 from the paper.
- The folder `other_experiments` contain scripts for generating the figures in the papers appendix.