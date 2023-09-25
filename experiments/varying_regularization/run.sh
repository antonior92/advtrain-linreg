mkdir results
mkdir plots

# Add parent directory to path
export PYTHONPATH="$PYTHONPATH:../.."
export PATH="$PATH:../.." # Add parent directory to path

# ----  Evaluate ---- #
ROOTDIR=results
mkdir $ROOTDIR
for DSET in latent magic gaussian rff;
  do mkdir $ROOTDIR/$DSET
  for METHOD in  lasso; # ridge advtrain_l2 advtrain_linf
    do mkdir $ROOTDIR/$DSET/$METHOD
      for SEED in 0 1 2 3 4; # 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29;
        do python estimate.py --dset $DSET -o $ROOTDIR/$DSET/$METHOD/results_s"$SEED" -s $SEED -m $METHOD --n_alpha 40 --alpha_range 6
      done;
  done;
done

# ----  Plot ---- #
# Plot Train MSE
STYLE="../mystyle.mplsty figsize.mplsty "
for DSET in latent magic gaussian rff;
do
  python plot.py $ROOTDIR/$DSET/advtrain_l2  $ROOTDIR/$DSET/ridge --log_yscale  --plot_style $STYLE \
  --ylabel "Train. MSE" --labels "\$\ell_2\$-adv. train" ridge  --loc 'lower left' \
  --plot_a2max --save plots/train_mse_l2_$DSET.pdf
  python plot.py $ROOTDIR/$DSET/advtrain_linf $ROOTDIR/$DSET/lasso \
    --log_yscale  --plot_style $STYLE --x_min 1e-1 --x_max 5e5 --y_min 1e-24 --y_max 1e1 \
     --ylabel "Train. MSE" --labels  "\$\ell_\infty\$-adv. train" lasso --loc 'lower left'\
     --plot_ainfmax --save plots/train_mse_linf_$DSET.pdf
done


# Plot Test MSE
for DSET in latent magic gaussian rff;
do
  python plot.py $ROOTDIR/$DSET/advtrain_l2  $ROOTDIR/$DSET/ridge  --plot_style $STYLE \
  --ylabel "Test MSE" --labels "\$\ell_2\$-adv. train" ridge  --loc 'upper left' --fields mse_test \
  --y_min 0 --y_max 5 --no_ytickformater \
  --save plots/test_mse_l2_$DSET.pdf
  python plot.py $ROOTDIR/$DSET/advtrain_linf $ROOTDIR/$DSET/lasso \
    --plot_style $STYLE  \
     --ylabel "Test MSE" --labels  "\$\ell_\infty\$-adv. train" lasso --loc 'upper left' --fields mse_test \
     --y_min 0 --y_max 5 --no_ytickformater \
     --save plots/test_mse_linf_$DSET.pdf
done


# Plot Adv MSE
for DSET in latent magic gaussian rff;
do
  python plot.py $ROOTDIR/$DSET/advtrain_l2  $ROOTDIR/$DSET/ridge  --plot_style $STYLE \
  --ylabel "Adv. MSE" --labels "\$\ell_2\$-adv. train" ridge  --loc 'upper left' --fields advrisk-2.0-0.1000000000 \
  --y_min 0 --y_max 20 --no_ytickformater \
  --save plots/adv_mse_l2_$DSET.pdf
  python plot.py $ROOTDIR/$DSET/advtrain_linf $ROOTDIR/$DSET/lasso \
    --plot_style $STYLE  \
     --ylabel "Adv. MSE" --labels  "\$\ell_\infty\$-adv. train" lasso --loc 'upper left' --fields advrisk-inf-0.1000000000 \
     --y_min 0 --y_max 20 --no_ytickformater \
     --save plots/adv_mse_linf_$DSET.pdf
done
