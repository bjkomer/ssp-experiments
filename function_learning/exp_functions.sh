dim=256
ntiles=4
nbins=8
limit=5
nsamples=20000
pcsigma=0.75
for seed in 1 2 3 4 5
do
  for enc in ssp hex-ssp pc-gauss one-hot tile-coding legendre
  do
    for concat in 0 1 
    do
      for func in distance direction centroid
      do
        python function_learning.py --concatenate ${concat} --spatial-encoding ${enc} --seed ${seed} --dim ${dim} --n-samples ${nsamples} --limit ${limit} --function ${func} --n-bins ${nbins} --n-tiles ${ntiles} --pc-gauss-sigma ${pcsigma}
      done
    done
  done
done
