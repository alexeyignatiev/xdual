#!/usr/bin/env python3
#-*- coding:utf-8 -*-

from __future__ import print_function
from data import Data
from options import Options
import os
import resource
import sys
from xgbooster import XGBooster


def enumerate_all(options, xtype, xnum, smallest, usecld, usemhs, useumcs, prefix):
    # setting the right preferences
    options.xtype = xtype
    options.xnum = xnum
    options.reduce = 'lin'
    options.smallest = smallest
    options.usecld = usecld
    options.usemhs = usemhs
    options.useumcs = useumcs

    # reading all unique samples
    with open('../bench/fairml/compas/compas.samples', 'r') as fp:
        lines = fp.readlines()

    # timers and other variables
    times, calls = [], []
    xsize, exlen = [], []

    # doing everything incrementally is expensive;
    # let's restart the solver for every 10% of instances
    tested = set()
    for i, s in enumerate(lines):
        # enumerate explanations only for the first 10% of samples
        if i % (len(lines) / 10) == 0:
            xgb = XGBooster(options, from_model='../temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl')

            # encode it and save the encoding to another file
            xgb.encode()

        options.explain = [float(v.strip()) for v in s.split(',')]

        if tuple(options.explain) in tested:
            continue

        tested.add(tuple(options.explain))
        print(prefix, 'sample {0}: {1}'.format(i, ','.join(s.split(','))))

        # calling anchor
        timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        expls = xgb.explain(options.explain)

        timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
        times.append(timer)

        print(prefix, 'expls:', expls)
        print(prefix, 'nof x:', len(expls))
        print(prefix, 'timex: {0:.2f}'.format(timer))
        print(prefix, 'calls:', xgb.x.calls)
        print(prefix, 'Msz x:', max([len(x) for x in expls]))
        print(prefix, 'msz x:', min([len(x) for x in expls]))
        print(prefix, 'asz x: {0:.2f}'.format(sum([len(x) for x in expls]) / len(expls)))
        print('')

        calls.append(xgb.x.calls)
        xsize.append(sum([len(x) for x in expls]) / float(len(expls)))
        exlen.append(len(expls))

    print('')
    print('all samples:', len(lines))

    # reporting the time spent
    print('{0} total time: {1:.2f}'.format(prefix, sum(times)))
    print('{0} max time per instance: {1:.2f}'.format(prefix, max(times)))
    print('{0} min time per instance: {1:.2f}'.format(prefix, min(times)))
    print('{0} avg time per instance: {1:.2f}'.format(prefix, sum(times) / len(times)))
    print('{0} total oracle calls: {1}'.format(prefix, sum(calls)))
    print('{0} max oracle calls per instance: {1}'.format(prefix, max(calls)))
    print('{0} min oracle calls per instance: {1}'.format(prefix, min(calls)))
    print('{0} avg oracle calls per instance: {1:.2f}'.format(prefix, float(sum(calls)) / len(calls)))
    print('{0} avg number of explanations per instance: {1:.2f}'.format(prefix, float(sum(exlen)) / len(exlen)))
    print('{0} avg explanation size per instance: {1:.2f}'.format(prefix, float(sum(xsize)) / len(xsize)))
    print('')


if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    # subset-minimal
    enumerate_all(options, xtype='abductive', xnum=-1, smallest=False, usecld=False, usemhs=False, useumcs=False, prefix='mabd')
    # enumerate_all(options, xtype='contrastive', xnum=-1, smallest=False, usecld=False, usemhs=True, useumcs=False, prefix='mcon3')
    # enumerate_all(options, xtype='contrastive', xnum=-1, smallest=False, usecld=False, usemhs=True, useumcs=True, prefix='mcon4')
    # enumerate_all(options, xtype='contrastive', xnum=-1, smallest=False, usecld=False, usemhs=False, useumcs=False, prefix='mcon1')
    # enumerate_all(options, xtype='contrastive', xnum=-1, smallest=False, usecld=False, usemhs=False, useumcs=True, prefix='mcon2')
    # enumerate_all(options, xtype='contrastive', xnum=-1, smallest=False, usecld=True, usemhs=False, useumcs=False, prefix='mcon5')
    # enumerate_all(options, xtype='contrastive', xnum=-1, smallest=False, usecld=True, usemhs=False, useumcs=True, prefix='mcon6')

    # # cardinality-minimal
    # explain_all(options, xtype='abductive', xnum=1, smallest=True, prefix='sabd')
    # explain_all(options, xtype='contrastive', xnum=1, smallest=True, prefix='scon')
