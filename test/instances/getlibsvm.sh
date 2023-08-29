
problems=(a1a a2a a3a australian \
	breast-cancer cod-rna colon-cancer \
  diabetes fourclass w1a w2a YearPredictionMSD)
for f in $problems; do
 echo $f;
 curl "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/$f" -o $f.libsvm
done
