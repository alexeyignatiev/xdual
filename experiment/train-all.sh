#!/bin/sh

cd ../

# adult
./xdual.py -p --pfiles adult.csv,adult bench/anchor/adult/
./xdual.py -c -t -n 50 bench/anchor/adult/adult_data.csv

# lending
./xdual.py -p --pfiles lending.csv,lending bench/anchor/lending/
./xdual.py -c -t -n 50 bench/anchor/lending/lending_data.csv

# recidivism
./xdual.py -p --pfiles recidivism.csv,recidivism bench/anchor/recidivism/
./xdual.py -c -t -n 50 bench/anchor/recidivism/recidivism_data.csv

# compas
./xdual.py -p --pfiles compas.csv,compas bench/fairml/compas/
./xdual.py -c -t -n 50 bench/fairml/compas/compas_data.csv

# german
./xdual.py -p --pfiles german.csv,german bench/fairml/german/
./xdual.py -c -t -n 50 bench/fairml/german/german_data.csv

# spambase
./xdual.py -p --pfiles spam10.csv,spam10 bench/nlp/spam/
./xdual.py -c -t -n 50 bench/nlp/spam/spam10_data.csv
