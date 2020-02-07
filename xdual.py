#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## xdual.py
##

from __future__ import print_function
from data import Data
from options import Options
import os
import sys
from xgbooster import XGBooster, preprocess_dataset


def show_info():
    """
        Print info message.
    """

    print('c XDual: dual explanations for XGBoost models')
    print('')


if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    # showing head
    show_info()

    if (options.preprocess_categorical):
        preprocess_dataset(options.files[0], options.preprocess_categorical_files)
        exit()

    if options.files:
        xgb = None

        if options.train:
            data = Data(filename=options.files[0], mapfile=options.mapfile,
                    separator=options.separator,
                    use_categorical = options.use_categorical)

            xgb = XGBooster(options, from_data=data)
            train_accuracy, test_accuracy, model = xgb.train()

        # read a sample from options.explain
        if options.explain:
            options.explain = [float(v.strip()) for v in options.explain.split(',')]

        if options.encode:
            if not xgb:
                xgb = XGBooster(options, from_model=options.files[0])

            # encode it and save the encoding to another file
            xgb.encode(test_on=options.explain)

        if options.explain:
            if not xgb:
                # abduction-based approach requires an encoding
                xgb = XGBooster(options, from_encoding=options.files[0])

            # explain using anchor or the abduction-based approach
            expl = xgb.explain(options.explain)

            # here we take only first explanation if case enumeration was done
            if options.xnum != 1:
                expl = expl[0]
