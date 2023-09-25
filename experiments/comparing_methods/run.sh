mkdir plots
mkdir results

# Add parent directory to path
export PYTHONPATH="$PYTHONPATH:../.."
export PATH="$PATH:../.." # Add parent directory to path

#  ----  Evaluate ---- #
python estimate.py --n_features 400 1000 2000 4000 8000 --n_samples -1 -o results/magic.csv

#  ----  Plot ---- #
python plot.py -i results/magic.csv -o plots/magic_deltabar.pdf