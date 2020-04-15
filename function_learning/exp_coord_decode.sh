dim=256
ntiles=4
nbins=8
limit=5
nsamples=20000
for seed in 1 2 3 4 5
do
  for enc in ssp hex-ssp pc-gauss one-hot tile-coding legendre random
  do
    python coord_decode.py --spatial-encoding ${enc} --seed ${seed} --dim ${dim} --n-samples ${nsamples} --limit ${limit} --n-bins ${nbins} --n-tiles ${ntiles} --logdir coord_decode_function/exps_dim${dim}_limit${limit}_${nsamples}samples/${enc}_seed${seed}
  done
done
