#!/bin/sh

# explain all dataset one by one
./enumerate-adult.py -e smt -s z3 -c -v temp/adult_data/adult_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > adult-explained.log

./enumerate-lending.py -e smt -s z3 -c -v temp/lending_data/lending_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > lending-explained.log

./enumerate-recidivism.py -e smt -s z3 -c -v temp/recidivism_data/recidivism_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > recidivism-explained.log

./enumerate-compas.py -e smt -s z3 -c -v temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > compas-explained.log

./enumerate-german.py -e smt -s z3 -c -v temp/german_data/german_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > german-explained.log

./enumerate-spam10.py -e smt -s z3 -c -v temp/spam10_data/spam10_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > spam10-explained.log
