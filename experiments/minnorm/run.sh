mkdir results
mkdir plots

# Add parent directory to path
export PYTHONPATH="$PYTHONPATH:../.."
export PATH="$PATH:../.." # Add parent directory to path

#  ----  Evaluate ---- #
for DSET in latent magic gaussian rff;
do
  python evaluate.py --dset $DSET -o results/$DSET.csv
done;

# ----  Plot threshold, and MSE ---- #
for DSET in latent magic gaussian rff;
do
  python plot.py -i results/$DSET.csv -f plots
done;
