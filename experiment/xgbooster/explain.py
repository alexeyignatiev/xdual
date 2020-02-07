#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## explain.py
##

from __future__ import print_function
import numpy as np
import os
from pysat.examples.hitman import Hitman
from pysat.formula import IDPool
from pysmt.shortcuts import Solver
from pysmt.shortcuts import And, BOOL, Iff, Implies, Ite, Not, Or, Symbol
from pysmt.shortcuts import Equals, GT, Int, INT, LE, Plus, Real, REAL
import resource
from six.moves import range
import sys


class SMTExplainer(object):
    """
        An SMT-inspired minimal explanation extractor for XGBoost models.
    """

    def __init__(self, formula, intvs, imaps, ivars, feats, nof_classes,
            options, xgb):
        """
            Constructor.
        """

        self.feats = feats
        self.intvs = intvs
        self.imaps = imaps
        self.ivars = ivars
        self.nofcl = nof_classes
        self.optns = options
        self.idmgr = IDPool()

        # saving XGBooster
        self.xgb = xgb

        self.verbose = self.optns.verb
        self.oracle = Solver(name=options.solver)

        self.inps = []  # input (feature value) variables
        for f in self.xgb.extended_feature_names_as_array_strings:
            if '_' not in f:
                self.inps.append(Symbol(f, typename=REAL))
            else:
                self.inps.append(Symbol(f, typename=BOOL))

        self.outs = []  # output (class  score) variables
        for c in range(self.nofcl):
            self.outs.append(Symbol('class{0}_score'.format(c), typename=REAL))

        # theory
        self.oracle.add_assertion(formula)

        # current selector
        self.selv = None

        # save and use dual explanations whenever needed
        self.dualx = []

        # number of oracle calls involved
        self.calls = 0

    def prepare(self, sample):
        """
            Prepare the oracle for computing an explanation.
        """

        if self.selv:
            # disable the previous assumption if any
            self.oracle.add_assertion(Not(self.selv))

        # creating a fresh selector for a new sample
        sname = ','.join([str(v).strip() for v in sample])

        # the samples should not repeat; otherwise, they will be
        # inconsistent with the previously introduced selectors
        assert sname not in self.idmgr.obj2id, 'this sample has been considered before (sample {0})'.format(self.idmgr.id(sname))
        self.selv = Symbol('sample{0}_selv'.format(self.idmgr.id(sname)), typename=BOOL)

        self.rhypos = []  # relaxed hypotheses

        # transformed sample
        self.sample = list(self.xgb.transform(sample)[0])

        self.sel2fid = {}  # selectors to original feature ids
        self.sel2vid = {}  # selectors to categorical feature ids

        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, self.sample), 1):
            feat = inp.symbol_name().split('_')[0]
            selv = Symbol('selv_{0}'.format(feat))
            val = float(val)

            self.rhypos.append(selv)
            if selv not in self.sel2fid:
                self.sel2fid[selv] = int(feat[1:])
                self.sel2vid[selv] = [i - 1]
            else:
                self.sel2vid[selv].append(i - 1)

        # adding relaxed hypotheses to the oracle
        if not self.intvs:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                if '_' not in inp.symbol_name():
                    hypo = Implies(self.selv, Implies(sel, Equals(inp, Real(float(val)))))
                else:
                    hypo = Implies(self.selv, Implies(sel, inp if val else Not(inp)))

                self.oracle.add_assertion(hypo)
        else:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                inp = inp.symbol_name()
                # determining the right interval and the corresponding variable
                for ub, fvar in zip(self.intvs[inp], self.ivars[inp]):
                    if ub == '+' or val < ub:
                        hypo = Implies(self.selv, Implies(sel, fvar))
                        break

                self.oracle.add_assertion(hypo)

        # in case of categorical data, there are selector duplicates
        # and we need to remove them
        self.rhypos = sorted(set(self.rhypos), key=lambda x: int(x.symbol_name()[6:]))

        # propagating the true observation
        if self.oracle.solve([self.selv] + self.rhypos):
            model = self.oracle.get_model()
        else:
            assert 0, 'Formula is unsatisfiable under given assumptions'

        # choosing the maximum
        outvals = [float(model.get_py_value(o)) for o in self.outs]
        maxoval = max(zip(outvals, range(len(outvals))))

        # correct class id (corresponds to the maximum computed)
        self.out_id = maxoval[1]
        self.output = self.xgb.target_name[self.out_id]

        # forcing a misclassification, i.e. a wrong observation
        disj = []
        for i in range(len(self.outs)):
            if i != self.out_id:
                disj.append(GT(self.outs[i], self.outs[self.out_id]))
        self.oracle.add_assertion(Implies(self.selv, Or(disj)))

        if self.verbose:
            inpvals = self.xgb.readable_sample(sample)

            self.preamble = []
            for f, v in zip(self.xgb.feature_names, inpvals):
                if f not in v:
                    self.preamble.append('{0} = {1}'.format(f, v))
                else:
                    self.preamble.append(v)

            print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

    def explain(self, sample, smallest):
        """
            Hypotheses minimization.
        """

        # reinitializing the number of used oracle calls
        # 1 because of the initial call checking the entailment
        self.calls = 1

        # adapt the solver to deal with the current sample
        self.prepare(sample)

        # saving external explanation to be minimized further
        self.to_consider = [True for h in self.rhypos]

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # if satisfiable, then the observation is not implied by the hypotheses
        if self.oracle.solve([self.selv] + [h for h, c in zip(self.rhypos, self.to_consider) if c]):
            print('  no implication!')
            print(self.oracle.get_model())
            sys.exit(1)

        if self.optns.xtype == 'abductive':
            # abductive explanations => MUS computation and enumeration
            if not smallest and self.optns.xnum == 1:
                expls = [self.compute_minimal_abductive()]
            else:
                expls = self.enumerate_abductive(smallest=smallest)
        else:  # contrastive explanations => MCS enumeration
            if self.optns.usemhs:
                expls = self.enumerate_contrastive()
            else:
                if not smallest:
                    expls = self.enumerate_minimal_contrastive()
                else:
                    # expls = self.enumerate_smallest_contrastive()
                    expls = self.enumerate_contrastive()

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        expls = list(map(lambda expl: sorted([self.sel2fid[h] for h in expl]), expls))

        if self.dualx:
            self.dualx = list(map(lambda expl: sorted([self.sel2fid[h] for h in expl]), self.dualx))

        if self.verbose:
            if expls[0] != None:
                for expl in expls:
                    preamble = [self.preamble[i] for i in expl]
                    if self.optns.xtype == 'abductive':
                        print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(preamble), self.xgb.target_name[self.out_id]))
                    else:
                        print('  explanation: "IF NOT {0} THEN NOT {1}"'.format(' AND NOT '.join(preamble), self.xgb.target_name[self.out_id]))
                    print('  # hypos left:', len(expl))

            print('  time: {0:.2f}'.format(self.time))

        # here we return the last computed explanation
        return expls

    def compute_minimal_abductive(self):
        """
            Compute any subset-minimal explanation.
        """

        i = 0

        # filtering out unnecessary features if external explanation is given
        rhypos = [h for h, c in zip(self.rhypos, self.to_consider) if c]

        # simple deletion-based linear search
        while i < len(rhypos):
            to_test = rhypos[:i] + rhypos[(i + 1):]

            self.calls += 1
            if self.oracle.solve([self.selv] + to_test):
                i += 1
            else:
                rhypos = to_test

        return rhypos

    def enumerate_minimal_contrastive(self):
        """
            Compute a subset-minimal contrastive explanation.
        """

        def _overapprox():
            model = self.oracle.get_model()

            for sel in self.rhypos:
                if int(model.get_py_value(sel)) > 0:
                    # soft clauses contain positive literals
                    # so if var is true then the clause is satisfied
                    self.ss_assumps.append(sel)
                else:
                    self.setd.append(sel)

        def _compute():
            i = 0
            while i < len(self.setd):
                if self.optns.usecld:
                    _do_cld_check(self.setd[i:])
                    i = 0

                if self.setd:
                    # it may be empty after the clause D check

                    self.calls += 1
                    self.ss_assumps.append(self.setd[i])
                    if not self.oracle.solve([self.selv] + self.ss_assumps + self.bb_assumps):
                        self.ss_assumps.pop()
                        self.bb_assumps.append(Not(self.setd[i]))

                i += 1

        def _do_cld_check(cld):
            self.cldid += 1
            sel = Symbol('{0}_{1}'.format(self.selv.symbol_name(), self.cldid))
            cld.append(Not(sel))

            # adding clause D
            self.oracle.add_assertion(Or(cld))
            self.ss_assumps.append(sel)

            self.setd = []
            st = self.oracle.solve([self.selv] + self.ss_assumps + self.bb_assumps)

            self.ss_assumps.pop()  # removing clause D assumption
            if st == True:
                model = self.oracle.get_model()

                for l in cld[:-1]:
                    # filtering all satisfied literals
                    if int(model.get_py_value(l)) > 0:
                        self.ss_assumps.append(l)
                    else:
                        self.setd.append(l)
            else:
                # clause D is unsatisfiable => all literals are backbones
                self.bb_assumps.extend([Not(l) for l in cld[:-1]])

            # deactivating clause D
            self.oracle.add_assertion(Not(sel))

        # sets of selectors to work with
        self.cldid = 0
        expls = []

        # detect and block unit-size MCSes immediately
        if self.optns.unitmcs:
            for i, hypo in enumerate(self.rhypos):
                self.calls += 1
                if self.oracle.solve([self.selv] + self.rhypos[:i] + self.rhypos[(i + 1):]):
                    expls.append([hypo])

                    if len(expls) != self.optns.xnum:
                        self.oracle.add_assertion(Or([Not(self.selv), hypo]))
                    else:
                        break

        self.calls += 1
        while self.oracle.solve([self.selv]):
            self.ss_assumps, self.bb_assumps, self.setd = [], [], []
            _overapprox()
            _compute()

            expl = [list(f.get_free_variables())[0] for f in self.bb_assumps]
            expls.append(expl)

            if len(expls) == self.optns.xnum:
                break

            self.oracle.add_assertion(Or([Not(self.selv)] + expl))
            self.calls += 1

        self.calls += self.cldid
        return expls if expls else [None]

    def enumerate_abductive(self, smallest=True):
        """
            Compute a cardinality-minimal explanation.
        """

        # result
        expls = []

        # just in case, let's save dual (contrastive) explanations
        self.dualx = []

        with Hitman(bootstrap_with=[[i for i in range(len(self.rhypos)) if self.to_consider[i]]], htype='sorted' if smallest else 'lbx') as hitman:
            # computing unit-size MCSes
            for i, hypo in enumerate(self.rhypos):
                if self.to_consider[i] == False:
                    continue

                self.calls += 1
                if self.oracle.solve([self.selv] + self.rhypos[:i] + self.rhypos[(i + 1):]):
                    hitman.hit([i])
                    self.dualx.append([self.rhypos[i]])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 1:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if self.oracle.solve([self.selv] + [self.rhypos[i] for i in hset]):
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(range(len(self.rhypos))).difference(set(hset)))

                    model = self.oracle.get_model()
                    for h in removed:
                        i = self.sel2fid[self.rhypos[h]]
                        if '_' not in self.inps[i].symbol_name():
                            # feature variable and its expected value
                            var, exp = self.inps[i], self.sample[i]

                            # true value
                            true_val = float(model.get_py_value(var))

                            if not exp - 0.001 <= true_val <= exp + 0.001:
                                unsatisfied.append(h)
                            else:
                                hset.append(h)
                        else:
                            for vid in self.sel2vid[self.rhypos[h]]:
                                var, exp = self.inps[vid], int(self.sample[vid])

                                # true value
                                true_val = int(model.get_py_value(var))

                                if exp != true_val:
                                    unsatisfied.append(h)
                                    break
                            else:
                                hset.append(h)

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        self.calls += 1
                        if self.oracle.solve([self.selv] + [self.rhypos[i] for i in hset] + [self.rhypos[h]]):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.verbose > 1:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)

                    self.dualx.append([self.rhypos[i] for i in to_hit])
                else:
                    if self.verbose > 1:
                        print('expl:', hset)

                    expl = [self.rhypos[i] for i in hset]
                    expls.append(expl)

                    if len(expls) != self.optns.xnum:
                        hitman.block(hset)
                    else:
                        break

        return expls

    def enumerate_smallest_contrastive(self):
        """
            Compute a cardinality-minimal contrastive explanation.
        """

        # result
        expls = []

        # computing unit-size MUSes
        muses = set([])
        for hypo in self.rhypos:
            self.calls += 1
            if not self.oracle.solve([self.selv, hypo]):
                muses.add(hypo)

        # we are going to discard unit-size MUSes from consideration
        rhypos = set(self.rhypos).difference(muses)

        # introducing interer cost literals for rhypos
        costlits = []
        for i, hypo in enumerate(rhypos):
            costlit = Symbol(name='costlit_{0}_{1}'.format(self.selv.symbol_name(), i), typename=INT)
            costlits.append(costlit)

            self.oracle.add_assertion(Ite(hypo, Equals(costlit, Int(0)), Equals(costlit, Int(1))))

        # main loop (linear search unsat-sat)
        i = 0
        while i < len(rhypos) and len(expls) != self.optns.xnum:
            # fresh selector for the current iteration
            sit = Symbol('iter_{0}_{1}'.format(self.selv.symbol_name(), i))

            # adding cardinality constraint
            self.oracle.add_assertion(Implies(sit, LE(Plus(costlits), Int(i))))

            # extracting explanations from MaxSAT models
            while self.oracle.solve([self.selv, sit]):
                self.calls += 1
                model = self.oracle.get_model()

                expl = []
                for hypo in rhypos:
                    if int(model.get_py_value(hypo)) == 0:
                        expl.append(hypo)

                # each MCS contains all unit-size MUSes
                expls.append(list(muses) + expl)

                # either stop or add a blocking clause
                if len(expls) != self.optns.xnum:
                    self.oracle.add_assertion(Implies(self.selv, Or(expl)))
                else:
                    break

            i += 1
            self.calls += 1

        return expls

    def enumerate_contrastive(self, smallest=True):
        """
            Compute a cardinality-minimal contrastive explanation.
        """

        # core extraction is done via calling Z3's internal API
        assert self.optns.solver == 'z3', 'This procedure requires Z3'

        # result
        expls = []

        # just in case, let's save dual (abductive) explanations
        self.dualx = []

        # mapping from hypothesis variables to their indices
        hmap = {h: i for i, h in enumerate(self.rhypos)}

        # mapping from internal Z3 variable into variables of PySMT
        vmap = {self.oracle.converter.convert(v): v for v in self.rhypos}
        vmap[self.oracle.converter.convert(self.selv)] = None

        def _get_core():
            core = self.oracle.z3.unsat_core()
            return sorted(filter(lambda x: x != None, map(lambda x: vmap[x], core)), key=lambda x: int(x.symbol_name()[6:]))

        def _do_trimming(core):
            for i in range(self.optns.trim):
                self.calls += 1
                self.oracle.solve([self.selv] + core)
                new_core = _get_core()
                if len(core) == len(new_core):
                    break
            return new_core

        def _reduce_lin(core):
            def _assump_needed(a):
                if len(to_test) > 1:
                    to_test.remove(a)
                    self.calls += 1
                    if not self.oracle.solve([self.selv] + list(to_test)):
                        return False
                    to_test.add(a)
                    return True
                else:
                    return True
            to_test = set(core)
            return list(filter(lambda a: _assump_needed(a), core))

        def _reduce_qxp(core):
            coex = core[:]
            filt_sz = len(coex) / 2.0
            while filt_sz >= 1:
                i = 0
                while i < len(coex):
                    to_test = coex[:i] + coex[(i + int(filt_sz)):]
                    self.calls += 1
                    if to_test and not self.oracle.solve([self.selv] + to_test):
                        # assumps are not needed
                        coex = to_test
                    else:
                        # assumps are needed => check the next chunk
                        i += int(filt_sz)
                # decreasing size of the set to filter
                filt_sz /= 2.0
                if filt_sz > len(coex) / 2.0:
                    # next size is too large => make it smaller
                    filt_sz = len(coex) / 2.0
            return coex

        def _reduce_coex(core):
            if self.optns.reduce == 'lin':
                return _reduce_lin(core)
            else:  # qxp
                return _reduce_qxp(core)

        with Hitman(bootstrap_with=[[i for i in range(len(self.rhypos)) if self.to_consider[i]]], htype='sorted' if smallest else 'lbx') as hitman:
            # computing unit-size MUSes
            for i, hypo in enumerate(self.rhypos):
                if self.to_consider[i] == False:
                    continue

                self.calls += 1
                if not self.oracle.solve([self.selv, self.rhypos[i]]):
                    hitman.hit([i])
                    self.dualx.append([self.rhypos[i]])
                elif self.optns.unitmcs:
                    self.calls += 1
                    if self.oracle.solve([self.selv] + self.rhypos[:i] + self.rhypos[(i + 1):]):
                        # this is a unit-size MCS => block immediately
                        hitman.block([i])
                        expls.append([self.rhypos[i]])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 1:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if not self.oracle.solve([self.selv] + [self.rhypos[h] for h in list(set(range(len(self.rhypos))).difference(set(hset)))]):
                    to_hit = _get_core()

                    if len(to_hit) > 1 and self.optns.trim:
                        to_hit = _do_trimming(to_hit)

                    if len(to_hit) > 1 and self.optns.reduce != 'none':
                        to_hit = _reduce_coex(to_hit)

                    self.dualx.append(to_hit)
                    to_hit = [hmap[h] for h in to_hit]

                    if self.verbose > 1:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    if self.verbose > 1:
                        print('expl:', hset)

                    expl = [self.rhypos[i] for i in hset]
                    expls.append(expl)

                    if len(expls) != self.optns.xnum:
                        hitman.block(hset)
                    else:
                        break

        return expls
