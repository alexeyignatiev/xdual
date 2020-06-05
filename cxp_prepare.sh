#!/bin/sh

# explain all images in experiments

./xdual.py -p --pfiles real_fake_20000_16.csv,real_fake_20000_16  ./bench/gan/

./xdual.py -c -t -n 100 --maxdepth 6 bench/gan/real_fake_20000_16_data.csv

./xdual.py -p --pfiles digits_3_5_20000_16.csv,digits_3_5_20000_16  ./bench/partition/

./xdual.py -c -t -n 50 --maxdepth 3 bench/partition/digits_3_5_20000_16_data.csv



