mkdir plots

# Add parent directory to path
export PYTHONPATH="$PYTHONPATH:.."
export PATH="$PATH:.." # Add parent directory to path

#  ----  Evaluate and plot---- #
python plot_thresholds.py --save plots
python varying_regularization.py --save plots/randomproj_l2attack_ --n_reps 5 --ord 2 --n_alphas 20
python varying_regularization.py --save plots/randomproj_linfattack_ --n_reps 5 --ord inf --n_alphas 20
