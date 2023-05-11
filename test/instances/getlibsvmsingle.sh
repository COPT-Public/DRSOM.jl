f=$1
echo $f
curl "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/$f" -o $f.libsvm
