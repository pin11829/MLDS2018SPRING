cat model4/ckpt.tar.gz.part* > model4/ckpt.tar.gz
tar zxvf model4/ckpt.tar.gz
mv ckpt/* model4/
PYTHONPATH=. python3 model4/test.py $1 $2
