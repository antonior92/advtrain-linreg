mkdir results
mkdir plots

# Add parent directory to path
export PYTHONPATH="$PYTHONPATH:.."
export PATH="$PATH:.." # Add parent directory to path


#  ----  Evaluate ---- #
for M in 300 1000 2000 4000 8000 16000;
do
  for SEED in 0 1 2 3 4;
  do
    python estimate.py -o results2/magic_m"$M"_seed"$SEED" -m $M -r $SEED --grid 100
  done;
done;

#  ---- Merge ---- #
python merge.py

#  ---- Plot ---- #
python plot.py


# NOTE: Comparing with the original paper:
# running the script WEBSITE/genomic.prediction.r
# We can extract the metric (1-R^2).
#    = mean((test.pheno$HET_2- pred.rr[,1])^2) / mean((test.pheno$HET_2- mean(test.pheno$HET_2))^2)
# Which we can compute there and
# Notice that this is the metric we have been plotting in merge.py
# The value obtained there is:
# - for 454/50 train/test split is:
#           ridge = 0.73784, lasso = 0.519179, elnet = 0.5166093
# - for 254/250 train/test split is:
#           ridge = 0.0.7933152, lasso = 0.5268, elnet = 0.5308153
# OBS: this may be different for different seeds...
# Since the cross validation and test set depend on the seed
# Generate multiple runs