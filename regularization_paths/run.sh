mkdir plots

# Add parent directory to path
export PYTHONPATH="$PYTHONPATH:.."
export PATH="$PATH:.." # Add parent directory to path

# ---- Diabetes example ---- #
python evaluate.py --save plots
# ---- Gaussian ---- #
python evaluate.py --save plots --dset gaussian
