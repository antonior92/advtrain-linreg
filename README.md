# Adversarially-trained linear regression

Companion code to the [paper](./paper.pdf)
```
Regularization properties of adversarially-trained linear regression
Antônio H Ribeiro, Dave Zachariah, Francis Bach, Thomas B. Sch\"on
NeurIPS 2023 (spotlight)
```


- The folder experiments contains the scripts to run the experiments we show in the paper. 
  Each experiment is in a separate folder. Some experiments are run in two stages, a script 
  usually called `evaluate.py` (or `estimate.py`) first populate a folder `results/` with 
  the results of the experiments and a script `plot.py` read the results and generate 
  the plots, saving them on `plots/` folder. See each individual experiment  folders for details. 
- Execute `run.sh` in the current directory to generate all the plots. 
- See `requirements.txt` for the required Python packages.


