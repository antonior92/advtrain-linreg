# Description: Script to run all experiments in the paper

### Download data ###
wget http://mtweb.cs.ucl.ac.uk/mus/www/MAGICdiverse/MAGIC_diverse_FILES/BASIC_GWAS.tar.gz
tar -xvf BASIC_GWAS.tar.gz

### Evaluate minimum norm interpolator ###
# ---- Plot Fig. 2 and S4, S5, S6 ---- #
( cd experiments/minnorm && sh run.sh )

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
