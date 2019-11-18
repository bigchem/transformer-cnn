#!/bin/bash 

SMI=$1;

mkdir test 
cd test 
rm -rf * 

cp ../embeddings.npy . 
sed '1d' $SMI | shuf | shuf > data

N=`wc -l $SMI | cut -d" " -f1`;
P=$(($N/5))
L=$(($N-$P*4))

head -n $P data > p1.csv 
head -n $((2*$P)) data | tail -n $P > p2.csv 
head -n $((3*$P)) data | tail -n $P > p3.csv 
head -n $((4*$P)) data | tail -n $P > p4.csv 
tail -n $L data  > p5.csv 

cat p2.csv p3.csv p4.csv p5.csv > t1.csv
cat p1.csv p3.csv p4.csv p5.csv > t2.csv 
cat p1.csv p2.csv p4.csv p5.csv > t3.csv
cat p1.csv p2.csv p3.csv p5.csv > t4.csv 
cat p1.csv p2.csv p3.csv p4.csv > t5.csv 

sed -e "s/train.csv/t1.csv/g; s/test.csv/p1.csv/g; s/MODE/True/g;" ../config-cv.cfg > train1.cfg
sed -e "s/train.csv/t2.csv/g; s/test.csv/p2.csv/g; s/MODE/True/g;" ../config-cv.cfg > train2.cfg
sed -e "s/train.csv/t3.csv/g; s/test.csv/p3.csv/g; s/MODE/True/g;" ../config-cv.cfg > train3.cfg
sed -e "s/train.csv/t4.csv/g; s/test.csv/p4.csv/g; s/MODE/True/g;" ../config-cv.cfg > train4.cfg
sed -e "s/train.csv/t5.csv/g; s/test.csv/p5.csv/g; s/MODE/True/g;" ../config-cv.cfg > train5.cfg

sed -e "s/train.csv/t1.csv/g; s/test.csv/p1.csv/g; s/MODE/False/g;" ../config-cv.cfg > pred1.cfg
sed -e "s/train.csv/t2.csv/g; s/test.csv/p2.csv/g; s/MODE/False/g;" ../config-cv.cfg > pred2.cfg
sed -e "s/train.csv/t3.csv/g; s/test.csv/p3.csv/g; s/MODE/False/g;" ../config-cv.cfg > pred3.cfg
sed -e "s/train.csv/t4.csv/g; s/test.csv/p4.csv/g; s/MODE/False/g;" ../config-cv.cfg > pred4.cfg
sed -e "s/train.csv/t5.csv/g; s/test.csv/p5.csv/g; s/MODE/False/g;" ../config-cv.cfg > pred5.cfg

python3 ../transformer-cnn.py train1.cfg 
python3 ../transformer-cnn.py pred1.cfg; sed '1d' r > r1

python3 ../transformer-cnn.py train2.cfg 
rm -f r; python3 ../transformer-cnn.py pred2.cfg; sed '1d' r > r2

python3 ../transformer-cnn.py train3.cfg 
rm -f r; python3 ../transformer-cnn.py pred3.cfg; sed '1d' r > r3

python3 ../transformer-cnn.py train4.cfg 
rm -f r ; python3 ../transformer-cnn.py pred4.cfg; sed '1d' r > r4

python3 ../transformer-cnn.py train5.cfg 
rm -f r ; python3 ../transformer-cnn.py pred5.cfg; sed '1d' r > r5

cut -d"," p1.csv -f2 > e1
cut -d"," p2.csv -f2 > e2
cut -d"," p3.csv -f2 > e3
cut -d"," p4.csv -f2 > e4
cut -d"," p5.csv -f2 > e5

cat e1 e2 e3 e4 e5 > e 
cat r1 r2 r3 r4 r5 > r 

paste e r > results

python3 ../q2.py results
