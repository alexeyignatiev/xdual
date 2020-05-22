#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## encode.py
##
##  Created on: Dec 7, 2018
##      Author: Alexey Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

#
#==============================================================================
from __future__ import print_function
import collections

from pysat.formula import IDPool
from pysmt.smtlib.parser import SmtLibParser
from pysmt.shortcuts import And, BOOL, Iff, Implies, Not, Or, Symbol, get_model
from pysmt.shortcuts import Equals, ExactlyOne, LT, Plus, REAL, Real, write_smtlib
from .tree import TreeEnsemble, scores_tree, walk_tree
import six
from six.moves import range
from ortools.sat.python import cp_model
import numpy  as np
import resource
from PIL import Image, ImageOps   
from math import fabs 
#from lime_wrap import lime_call
#from anchor_wrap import anchor_call
from shap_wrap import shap_call
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale

import random
try:  # for Python2
    from cStringIO import StringIO
except ImportError:  # for Python3
    from io import StringIO

import matplotlib.pyplot as plt
import os

EXTRA_FIX_TAG  = "extra_fix"
ONE_HOT_TAG  = "one_hot"
COST_TAG = "cost"
BLOCK_TAG = "block"
SUM_COST_TAG  = "sum_cost"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
#==============================================================================
class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, encoder, hypos_feat):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.encoder = encoder
        self.hypos_feat = hypos_feat
        self.__solution_count = 0
        self.solution_collect = []
        self.sample_internal_collect = []
        self.sample_internal_feat_collect = []
        self.block_collect = []
        self.block_str_collect = []
        self.tracking = {}        
        for i,v in  sorted(self.encoder.var2ids.items()):
            self.tracking[v] = []

    def on_solution_callback(self):
        self.__solution_count += 1
        solution, sample_internal, sample_internal_feat, block, block_str = self.encoder.get_solution(self, self.hypos_feat)
        self.sample_internal_collect.append(sample_internal)
        self.sample_internal_feat_collect.append(sample_internal_feat)
        self.block_collect.append(block)
        self.block_str_collect.append(block_str)        
        self.solution_collect.append(solution)                
        print(self.__solution_count, solution)
        #for v in  self.hypos_feat:
        #    print("name=", v, self.Value(v))
        #exit()
        if (False):
            for i,v in  sorted(self.encoder.var2ids.items()):
                #print("name=",i,v, self.Value(v))
                self.tracking[v].append(self.Value(v))
            if (self.__solution_count > 1):
                for i,v in  sorted(self.encoder.var2ids.items()):
                    if (self.tracking[v][-2] != self.tracking[v][-1]):
                        print("name=",i,v, self.Value(v), self.tracking[v])
                        #exit()
            
        #print(self.tracking[v])

    def solution_count(self):
        return self.__solution_count


class OREncoder(object):
    """
        Encoder of XGBoost tree ensembles into SMT.
    """

    def __init__(self, model, feats, nof_classes, xgb, from_file=None):
        """
            Constructor.
        """

        self.or_model = cp_model.CpModel()
        self.solver =  cp_model.CpSolver()

        self.model = model
        self.feats = {f: i for i, f in enumerate(feats)}
        self.nofcl = nof_classes
        self.idmgr = IDPool()
        self.optns = xgb.options
        self.used_f_vars = []
        # xgbooster will also be needed
        self.xgb = xgb
        self.verbose = self.optns.verb        
        
        self.vpool = 0
        self.var2ids = {}
        self.assign2tvars = {}
        self.reuse_t = 0
        
        # for interval-based encoding
        self.intvs, self.imaps, self.ivars = None, None, None

        if from_file:
            self.load_from(from_file)
            
    def get_feature_bool_var(self, model, var):
        if var not in self.var2ids:
            self.var2ids[var] = model.NewBoolVar(name = var)
            return self.var2ids[var]
        else:
            return self.var2ids[var]
            

    
    def traverse(self, tree, tvars, prefix=[]):
        """
            Traverse a tree and encode each node.
        """

        if tree.children:
            pos, neg = self.encode_node(tree)

            self.traverse(tree.children[0], tvars, prefix + [pos])
            self.traverse(tree.children[1], tvars, prefix + [neg])
        else:  # leaf node
            #print(tree.values, self.round_value(tree.values))
            #exit()
            v = self.round_value(tree.values)
            #print([tvar], v)

            if prefix:
                str_prefix = []
                for p in prefix:
                    str_prefix.append(p.__str__())
                    if ('not' in p.__str__()):
                        continue
                    self.used_f_vars.append(p.__str__())
                    
                if str(str_prefix) in self.assign2tvars:
                    tvar = self.assign2tvars[str(str_prefix) ]
                    #exit()
                    #print("reuse", str_prefix, tvar, v)
                    self.reuse_t = self.reuse_t + 1
                else:
                    tvar =  self.get_feature_bool_var(self.or_model, 'tvar_{0}_{1}'.format( self.working_tree, len(tvars)))
                    self.assign2tvars[str(str_prefix) ] = tvar
                    #print("new", str_prefix, tvar, v)

                #temp_var = self.get_feature_bool_var('temp_var_{0}_{1}_{2}'.format(self.working_tree, tree.id, len(tvars)))
                c1 = self.or_model.AddBoolAnd(prefix).OnlyEnforceIf(tvar)
                self.enc[self.working_tree].append([c1, " And({}) -> {} ".format(str_prefix, tvar)])

                #c2 = self.or_model.AddBoolOr([x.Not() for x in prefix]).OnlyEnforceIf(temp_var.Not())
                #self.enc[self.working_tree].append([c2, " Not And({}) -> ({}) ".format(str_prefix, temp_var.Not())])
                
                #impl = self.or_model.AddImplication(temp_var, tvar)
                #self.enc[self.working_tree].append([impl, "imply And({}) -> {} = {} ".format(str_prefix, tvar, v)])
            else:
                tvar =  self.get_feature_bool_var(self.or_model, 'tvar_{0}_{1}'.format( self.working_tree, len(tvars)))                
                self.enc[self.working_tree].append([tvar, "{} = {} ".format(tvar, v)] )

            tvars.append([tvar, v])
            #print(tvars, impl, v)
            #exit()

    def round_value(self, v, digits = 100000):
        v = (np.rint(v*digits)).astype(int)
        return v
        
    def encode_node(self, node):
        """
            Encode a node of a tree.
        """

        if '_' not in node.name:            
            assert(False)
# 
#             # continuous features => expecting an upper bound
#             # feature and its upper bound (value)
#             f, v = node.name, node.threshold
# 
#             existing = True if tuple([f, v]) in self.idmgr.obj2id else False
#             vid = self.idmgr.id(tuple([f, v]))
#             bv = self.or_model.NewBoolVar('bvar{0}'.format(vid))
# 
#             if not existing:
#                
#                 
#                 
#                 if self.intvs:
#                     d = self.imaps[f][v] + 1
#                     pos, neg = self.ivars[f][:d], self.ivars[f][d:]
#                     self.enc.append(Iff(bv, Or(pos) ))
#                     self.enc.append(Iff(Not(bv), Or(neg)))
#                 else:
#                     assert(False)
#                     fvar, fval = Symbol(f, typename=REAL), Real(v)
#                     self.enc.append(Iff(bv, LT(fvar, fval)))
# 
#             return bv, bv.Not()
        else:
            # all features are expected to be categorical and
            # encoded with one-hot encoding into Booleans
            # each node is expected to be of the form: f_i < 0.5
            bv = self.get_feature_bool_var(self.or_model, node.name)

            # left branch is positive,  i.e. bv is true
            # right branch is negative, i.e. bv is false
            return bv.Not(), bv

    def compute_intervals(self):
        """
            Traverse all trees in the ensemble and extract intervals for each
            feature.

            At this point, the method only works for numerical datasets!
        """

        def traverse_intervals(tree):
            """
                Auxiliary function. Recursive tree traversal.
            """

            if tree.children:
                f = tree.name
                v = tree.threshold
                self.intvs[f].add(v)

                traverse_intervals(tree.children[0])
                traverse_intervals(tree.children[1])

        # initializing the intervals
        self.intvs = {'f{0}'.format(i): set([]) for i in range(len(self.feats))}

        for tree in self.ensemble.trees:
            traverse_intervals(tree)

        # OK, we got all intervals; let's sort the values
        self.intvs = {f: sorted(self.intvs[f]) + ['+'] for f in six.iterkeys(self.intvs)}

        self.imaps, self.ivars = {}, {}
        for feat, intvs in six.iteritems(self.intvs):
            self.imaps[feat] = {}
            self.ivars[feat] = []
            for i, ub in enumerate(intvs):
                self.imaps[feat][ub] = i

                ivar = Symbol(name='{0}_intv{1}'.format(feat, i), typename=BOOL)
                self.ivars[feat].append(ivar)

    def encode(self):
        """
            Do the job.
        """

        self.enc = {}

        # getting a tree ensemble
        self.ensemble = TreeEnsemble(self.model,
                self.xgb.extended_feature_names_as_array_strings,
                nb_classes=self.nofcl)

        # self.ensemble.print_tree()

        # introducing class score variables
        csum = []
        #for j in range(self.nofcl):
            
        #    cvar = self.or_model.NewIntVar(lb = -large_number, ub = large_number, name = 'class{0}_score'.format(j))
        #    csum.append(tuple([cvar, []]))

        #print(cvar)
        # if targeting interval-based encoding,
        # traverse all trees and extract all possible intervals
        # for each feature
        if self.optns.encode == 'smtbool':
            assert(False)
            self.compute_intervals()


        if (self.nofcl != 2):
            print("ortools works for 2 class classification only in the current version. Please use smt option for multiple classes.")
            assert(self.nofcl == 2)
            
        # traversing and encoding each tree
        scores = {}
        k = 0
        for i, tree in enumerate(self.ensemble.trees):
            # getting class id
            clid = i % self.nofcl
            if (clid == 1):
                continue

            # encoding the tree
            tvars = []#self.or_model.NewIntVar(lb = -large_number, ub = large_number, name = 'tr{0}_score'.format(i + 1))
            self.working_tree = k
            self.enc[self.working_tree] = []
            self.traverse(tree, tvars, prefix=[])
            self.working_tree = None
            scores[k] = tvars
            k = k + 1

            # this tree contributes to class with clid
            #csum[clid][1].append(tvar)
        #print(self.enc)
        #print(self.or_model)
        #exit()
        #print(scores)
        # encoding the sums
        #sum_ex = cp_model.LinearExpr.ScalProd(vars, np.array(coeff).astype(int))        
        self.lin_vars = []
        self.lin_coef = []
        for i, score_tree in scores.items():
            #print(score_tree)
            # at most one
            vars =  [l[0] for l in score_tree]
            con = self.or_model.Add(sum(vars) == 1)
            self.enc[i].append([con, "Sum({})=1".format(vars)])
            #cvar, tvars = pair
            #self.enc.append(Equals(cvar, Plus(tvars)))
            for tvar, c  in score_tree:
                self.lin_vars.append(tvar)
                self.lin_coef.append(c)
        #print(lin_vars, lin_coef) 
        self.cost = cp_model.LinearExpr.ScalProd(self.lin_vars, np.array(self.lin_coef))        
        #c = self.or_model.Add(sum_ex > 0)
        #self.enc.append([c,"sum({}*{}) >0".format(self.lin_vars,self.lin_coef)])      
        # enforce exactly one of the feature values to be chosen
        # (for categorical features)
        if (False):
            categories = collections.defaultdict(lambda: [])
            self.categories_enc = collections.defaultdict(lambda: [])
            self.enc[ONE_HOT_TAG] = []    
            for i, f in enumerate(self.xgb.extended_feature_names_as_array_strings):
                if '_' in f:
                    
                    var = self.get_feature_bool_var(self.or_model, f)
                    var_tag = '{:04d}_'.format(i) + f.split('_')[0]
                    id = f.split('_')[0]                
                    categories[id].append(var)
                    self.categories_enc[var_tag].append(f)
    
                  
            for c, feats in six.iteritems(categories):
                con = self.or_model.Add(sum(feats) == 1)
                self.enc[ONE_HOT_TAG].append([con, "sum({}) = 1".format(feats)])

        # number of assertions
        nof_asserts = len(self.enc)

        # making conjunction
        #self.enc = And(self.enc)

        # number of variables
        #nof_vars = len(self.enc.get_free_variables())

        if self.optns.verb:
            print('encoding vars:', len(self.var2ids))
            print('encoding asserts:', nof_asserts)
    
        return self.enc, self.intvs, self.imaps, self.ivars
    def get_score(self, sample):
        csum = [[] for c in range(self.nofcl)]        
        # traversing all trees
        for i, tree in enumerate(self.ensemble.trees):
            # getting class id
            clid = i % self.nofcl

            # a score computed by the current tree
            score = scores_tree(tree, sample)

            # this tree contributes to class with clid
            csum[clid].append(score)

        # final scores for each class
        cscores = [sum(scores) for scores in csum]        
        return cscores, csum
    def copy_model(self, model):
        clean_model = cp_model.CpModel()
        clean_model._CpModel__model.CopyFrom(model._CpModel__model)
        clean_model._CpModel__constant_map.update(model._CpModel__constant_map)
        clean_model._CpModel__optional_constant_map.update(clean_model._CpModel__optional_constant_map)  
        return  clean_model     
    def get_solution(self, solver, hypos_feat):
        block = []
        block_str = []
        sample_internal = []
        sample_internal_feat = []
        cnt = 0
        for i,feat in  self.categories_enc.items():
            for j,f in enumerate(feat):
                if (f == 0):
                    sample_internal.append(int(f))
                    sample_internal_feat.append("{}: {}={}".format(cnt, f,0))    
                    cnt = cnt + 1   
                    continue
                if (solver.Value(self.get_feature_bool_var(self.or_model, f)) == 1):                          
                    sample_internal.append(1)
                    sample_internal_feat.append("{}: {}={}".format(cnt, f,1))
                else:
                    sample_internal.append(0)
                    sample_internal_feat.append("{}: {}={}".format(cnt, f,0))    
                    if (f in hypos_feat):
                        #print(f)
                        #print(hypos_feat)
                        #exit()
                        block.append(self.get_feature_bool_var(self.or_model, f))
                        block_str.append(self.get_feature_bool_var(self.or_model, f).__str__())                                
                cnt = cnt + 1       
        solution = list(self.xgb.transform_inverse(np.asarray(sample_internal))[0])
                                  
        return solution, sample_internal, sample_internal_feat, block, block_str 
   
    def used_categories(self):
        so_far_used_categories = collections.defaultdict(lambda: [])
        for i, f in enumerate(self.xgb.extended_feature_names_as_array_strings):
            if '_' in f:                                        
                id = f.split('_')[0]        
                if (f  in self.used_f_vars):        
                    so_far_used_categories[id].append(f)    
        return so_far_used_categories
    def test_sample(self, sample):
        """
            Check whether or not the encoding "predicts" the same class
            as the classifier given an input sample.
        """
        # first, compute the scores for all classes as would be
        # predicted by the classifier

        # score arrays computed for each class
        sample = sample[:-1]
        debug =  False
        reproduce =  True

        is_image = True
        is_text = False
        print("\n-->Starting enumeration:")
        if self.optns.verb:
            print('testing sample:', list(sample))

        sample_internal_o = list(self.xgb.transform(sample)[0])
        solution_o = (self.xgb.transform_inverse(np.asarray(sample_internal_o))[0])
        #print(solution_o)

        #print(sample_internal)

        cscores, csum = self.get_score(sample_internal_o)
    
        #print(cscores, len(csum[0]))

        # second, get the scores computed with the use of the encoding

        # asserting the sample
        hypos = []
        hypos_feat = []

        assert(not self.intvs)

        so_far_used_categories = self.used_categories()                        
        for i, fval in enumerate(sample_internal_o):
            feat, vid = self.xgb.transform_inverse_by_index(i)
            fid = self.feats[feat]
            
            #print(i, fval, feat, vid, fid)
            if vid == None:
                assert(False)
            else:
                f =  'f{0}_{1}'.format(fid, vid)    
                #if (fid < 124): 
                #    print(i, f, fval, sample_internal_o[i])               
                selv = Symbol('selv_{0}'.format(fid))
                #fvar = self.get_feature_bool_var(self.or_model, f)

                if int(fval) == 1:
                    fvar = self.get_feature_bool_var(self.or_model, f)
                    hypos.append(fvar)
                    hypos_feat.append(f)
                    #print( self.unused_categories)
                    #if not (f  in  self.used_f_vars):
                    #    id = f.split('_')[0]              
                    #    #print( so_far_used_categories[id])
                    #    if (len(so_far_used_categories[id]) == 0):
                    #        continue                           
                    self.used_f_vars.append(f)
                        #print(f, len(self.var2ids))
                    
        #exit()            

                #else:
                #    hypos.append(fvar.Not())

        ################# finishing encoding 
        self.enc[ONE_HOT_TAG] = []
        self.unused_categories = collections.defaultdict(lambda: [])
        self.used_categories = collections.defaultdict(lambda: [])
        for i, f in enumerate(self.xgb.extended_feature_names_as_array_strings):
            if '_' in f:                                        
                id = f.split('_')[0]        
                if (f  in self.used_f_vars):        
                    self.used_categories[id].append(f)
                else:
                    self.unused_categories[id].append(f)

        # copy last from unused to used
        for id, vars in self.unused_categories.items():        
            if (len(vars) == 0):
                continue            
            self.used_categories[id].append(vars[-1])
            self.used_f_vars.append(vars[-1])
            self.unused_categories[id] = self.unused_categories[id][:-1]
        for c, vars in six.iteritems(self.used_categories):
            if (len(vars) <= 1):
                #print(vars)
                if (len(vars) == 1):
                    self.get_feature_bool_var(self.or_model, vars[0]) 
                continue
            feats = [self.get_feature_bool_var(self.or_model, f) for f in vars]
            con = self.or_model.Add(sum(feats) == 1)
            self.enc[ONE_HOT_TAG].append([con, "sum({}) = 1".format(feats)])
            #print(vars)
        self.categories_enc = collections.defaultdict(lambda: [])
        #exit()
        for i, f in enumerate(self.xgb.extended_feature_names_as_array_strings):
            if '_' in f:
                var_tag = '{:06d}_'.format(i) + f.split('_')[0]
                id = f.split('_')[0]                
                v = int(id[1:])
                #print(v)
                #if (v < 124): 
                #    print(i, var_tag, f)
                if (f  in self.used_f_vars):                            
                    self.categories_enc[var_tag].append(f)
                    #if (v < 124): 
                    #    print("used")

                else:
                    #if (v < 124): 
                    #    print("unused")

                    #print(f.split('_')[1])
                    #if (solution_o[v] > 0):
                    #    print(f, solution_o[v])
                    self.categories_enc[var_tag].append(0)
        #print("hi") 
        #exit()

        if (debug):            
            print("total vars {}, reused tree vars {}".format(len(self.var2ids.items()), self.reuse_t))
            print("used input lits", len(np.unique(np.sort(np.asarray(self.used_f_vars)))))
        
        ################3 classification
        self.enc[BLOCK_TAG] = []
        self.enc[COST_TAG] = []

        if (cscores[0] > cscores[1]):
            con = self.or_model.Add(self.cost < 0)
            self.enc[COST_TAG] = [[con,"sum({}*{}) < 0".format(self.lin_vars,self.lin_coef)]]
            self.output = self.xgb.target_name[0]
        else:
            con = self.or_model.Add(self.cost >= 0)
            self.enc[COST_TAG] = [[con,"sum({}*{}) >= 0".format(self.lin_vars,self.lin_coef)]]
            self.output = self.xgb.target_name[1]

        if (reproduce):
            self.solver.parameters.num_search_workers = 8
        #self.solver.parameters.optimize_with_max_hs = True
        #self.solver.parameters.log_search_progress = True
        cnt = 0
        times = []

        ############ cost
        cvar = self.or_model.NewIntVar(lb = 0, ub = len(hypos), name = SUM_COST_TAG )
        self.or_model.Add(cvar == sum(hypos))
        objective = self.or_model.Maximize(cvar)
        timer1 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime        
        expls = []
        hits = collections.defaultdict(lambda: [])
        prev_obj = -1
        obj = -1
        xgimportance = np.asarray(self.xgb.model.feature_importances_)
        #print(len(sample_internal_o))
        importance = np.zeros(len(solution_o))
        

        
        if (reproduce):    
            map_text = {}
            if(is_image):
                side = int(np.sqrt(len(solution_o)))
    
                image_dir = "./image_temp/"
                ensure_dir(image_dir)
                if (self.optns.output == ''):
                    image_dir = "./image_temp/{}/".format(random.randint(0, 10000000))
                else:
                    image_dir = "./image_temp/{}/".format(self.xgb.options.output)
                    
                ensure_dir(image_dir)
            
                
                suf = "results_{}".format("")#"gan"#
                solution_o_r = solution_o.reshape((side,side))
                #rescaled = (255.0 / solution_o_r.max() * (solution_o_r - solution_o_r.min())).astype(np.uint8)        
                #im1 = Image.fromarray(rescaled)
                #im1 = im1.save("{}sample_{}.jpg".format(image_dir, suf))
    
                plt.close('all')
                fig = plt.figure()
                cbar = plt.imshow(solution_o_r/(np.max(solution_o_r)),  cmap='gray', interpolation = 'none')
                plt.colorbar()
                plt.delaxes(fig.axes[1])                
                plt.savefig("{}sample_heat_{}.jpg".format(image_dir,suf))  

                plt.close('all')
                fig = plt.figure()
                r = solution_o_r.copy()
                r[r <= 0] = -50

                r = r/(np.max(r))
                cbar = plt.imshow(r/(np.max(r)),  cmap='gray', interpolation = 'none')
                plt.colorbar()
                plt.delaxes(fig.axes[1])                
                plt.savefig("{}_non-zero_focus_sample_heat_{}.jpg".format(image_dir,suf))  

                print("starting shap")
                expl, scores = shap_call(self.xgb, sample=sample, nb_features_in_exp=side*side, getscores = True, silent = True)
                expl_image = np.zeros((side*side))
                expl_image[expl] = np.abs(scores)
                expl_image  = expl_image.reshape((side,side))
                print("done")

                plt.close('all')
                fig = plt.figure()
                cbar = plt.imshow(expl_image/(np.max(expl_image)), cmap='hot', interpolation = 'none')
                plt.colorbar() 
                plt.delaxes(fig.axes[1])                               
                plt.savefig("{}shap_features_{}.jpg".format(image_dir,suf))        
                
                          
                #exit()
                
                for i, f in enumerate(self.xgb.extended_feature_names_as_array_strings):
                    #print(i)
                    if '_' in f:
                        var_tag = int((f.split('_')[0])[1:])
                        importance[var_tag] = importance[var_tag] + xgimportance[i]                      
                #print(importance)
            
                importance_r = importance.reshape((side,side))
                plt.close('all')
                fig = plt.figure()                
                cbar = plt.imshow(importance_r/(np.max(importance_r)), cmap='hot', interpolation = 'none')
                plt.colorbar()
                plt.delaxes(fig.axes[1])                               

                plt.savefig("{}xgboost_features_{}.jpg".format(image_dir,suf))        
                #exit()
            if (is_text):
                text_dir = "./text_temp/"
                ensure_dir(text_dir)
                text_dir = "./text_temp/{}/".format(random.randint(0, 10000000))
                ensure_dir(text_dir)
                sz = len(sample)
                suf = "results"                            
       
                
                expl, scores = shap_call(self.xgb, sample=sample, nb_features_in_exp=sz, getscores = True)
                expl_text = np.zeros(sz)
                expl_text[expl] = np.abs(scores)
                expl_text = expl_text
                map_text['shap'] = minmax_scale(expl_text)
                         
                            
                expl = anchor_call(self.xgb, sample=sample, nb_features_in_exp=sz)
                expl_text = np.zeros(sz)
                expl_text[expl] = 1
                expl_text = expl_text
                map_text['anchor'] = minmax_scale(expl_text)

                expl, scores = lime_call(self.xgb, sample=sample, nb_features_in_exp=sz, getscores = True)
                expl_text = np.zeros(sz)
                expl_text[expl] = np.abs(scores)
                expl_text = expl_text
                map_text['lime'] = minmax_scale(expl_text)
                #exit()
                       
                
        max_mcses = self.xgb.options.xnum    
        prev_block = None
        while (True):
            #print("start1")                                 
            print("+{}".format(cnt+1),end = "")
            status = self.solver.Solve(self.or_model)   
            print("-",end = "")
#             print('Solve status: %s' % self.solver.StatusName(status))
#             print('Statistics')
#             print('  - conflicts : %i' % self.solver.NumConflicts())
#             print('  - branches  : %i' % self.solver.NumBranches())
#             print('  - wall time : %f s' % self.solver.WallTime())
# 
            #print(self.solver.ResponseStats())
#             exit()
            if (reproduce):
                if status == cp_model.OPTIMAL:
                    #print('{} Objective value = {}'.format(cnt, self.solver.ObjectiveValue()), end = " ")
                    if(prev_obj == -1):
                        prev_obj = self.solver.ObjectiveValue()
                        heat = np.zeros(len(solution_o))
                        heat_all = np.zeros(len(solution_o))
                        
                    obj = self.solver.ObjectiveValue()
            
            if status == cp_model.OPTIMAL:
                solution, sample_internal, sample_internal_feat, block, block_str = self.get_solution(self.solver, hypos_feat)
                scores, csum =  self.get_score(sample_internal)
                
                round_csum = []
                for ts in range(0, len(csum[0])):
                    round_csum.append(self.round_value(csum[0][ts]))

                score = self.round_value(scores[0])
                #for i,v in  sorted(self.var2ids.items()):
                #    print("name=",i,v, self.solver.Value(v))
                s= 0
                #print(round_csum)
                cnt_1 = 0
                for j, v in  enumerate(self.lin_vars):
                    s = s + int(self.solver.Value(self.lin_vars[j])*self.lin_coef[j])
                    if(self.solver.Value(self.lin_vars[j]) == 1):
                        val = self.solver.Value(self.lin_vars[j])
                        assert(abs(round_csum[cnt_1] -  self.lin_coef[j]) < 10)
                        cnt_1 = cnt_1 + 1
                solution_n = (self.xgb.transform_inverse(np.asarray(sample_internal))[0])     
                
                if (reproduce):
                    #print(cnt)

                    if  (cnt %100 == 0) or (obj < prev_obj):
                        index =  len(block) 
                        if (obj < prev_obj):
                            index =  len(prev_block)
                            
                        if (is_text) and (prev_block is not None):
                            #plt.close('all')
                            #fig = plt.figure(figsize=(16, 2))
                            heat_r = heat
                            #cbar = plt.imshow(heat_r/(np.max(heat_r)), cmap='hot', interpolation = 'none')
                            map_text['heat_{}'.format(index)] = minmax_scale(heat_r)
                            #print('heat_{}'.format(index), map_text['heat_{}'.format(index)])
                            if (obj < prev_obj):
                                heat =   np.zeros(len(solution_n)) + (solution_o != solution_n)
                                #print(heat_temp)
                                map_text['heat_{}'.format(len(block))] = minmax_scale(heat)
                                #print('heat_{}'.format(len(block)) , map_text['heat_{}'.format(len(block))])

                            #heat_all_r = (heat_all + heat)
                            #map_text['heat_all'] = minmax_scale(heat_all_r)

                            heatmaps = np.vstack(list(map_text.values()))
                            
                            plt.close('all')
                            fig = plt.figure(figsize=(16, 8))    
                            #print(heatmaps)               
                            plt.imshow(heatmaps, cmap='hot')
                            plt.colorbar() 
                            plt.delaxes(fig.axes[1])                               
                            ax1 = plt.axes()
                            #y_axis = ax1.axes.get_yaxis()
                            #y_axis.set_visible(False)                
                            plt.yticks(np.arange(len(heatmaps)), map_text.keys(),  fontsize=8)
                            plt.xticks(np.arange(sz), self.xgb.feature_names,  fontsize=8)
                            degrees = 90
                            plt.xticks(rotation=degrees)
                            ax1.set_xlabel("Features",  fontsize=12)         
                            my_colors = ['k']*sz
                            #print(my_colors)
                            for i in range(sz):
                                if (sample[i] > 0):
                                    my_colors[i] = 'g'
            
                            for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
                                ticklabel.set_color(tickcolor)                                                                         
                            plt.savefig("{}text_heats{}.jpg".format(text_dir,suf))
                            #map_text.pop("heat_all", None)
                            
                            #exit()
                        if(is_image):  

                            heat_r = heat.reshape((side,side))                        

                            #print(cnt, (normalize(heat_r)))
                            plt.close('all')
                            fig = plt.figure()                            
                            plt.imshow(heat_r/(np.max(heat_r)), cmap='hot', interpolation = 'none')
                            plt.colorbar()
                            plt.delaxes(fig.axes[1])                                                       
                            plt.savefig("{}heat_{}_{}.jpg".format(image_dir,suf, index ))
                            #print(heat_r)
                            #exit()
                            heat_all_r = (heat_all + heat).reshape((side,side))
                            plt.close('all')
                            fig = plt.figure()                            
                            plt.imshow(heat_all_r/(np.max(heat_all_r)), cmap='hot')
                            plt.colorbar()
                            plt.delaxes(fig.axes[1])                                                       
                            plt.savefig("{}heat_all_{}.jpg".format(image_dir,suf, ))
                            #if (obj < prev_obj):
                            #    exit()
                        #exit()

                #print(solution_o)
                #print(solution_n)
                
                #exit()
                heat = heat + (solution_o != solution_n)
                prev_block = block
                #print(heat)
                #exit()

                if (cnt > max_mcses):
                    break

                #print(obj, prev_obj)
                if (obj < prev_obj):
                    heat_all = heat_all + heat
                    heat = np.zeros(len(solution_n)) + (solution_o != solution_n)
                    prev_obj = obj

                    #print("step {}: {}, {}, truth = {}, {}".format(cnt, solution, s, score, block_str))
                    #print("step {}: {},  truth = {}, {}".format(cnt,  s, score, block_str))
                if np.sign(s) != np.sign(score):
                    print("A wrong MCS due to precision, skip it")
                else:
                    expl = [int((f.split('_')[0])[1:]) for f in block_str]
                    expls.append(expl)
            else:
                #print("UNSAT")
                break
            
            bound = int(self.solver.ObjectiveValue())

            b = self.or_model.Add(sum(block) >= 1)
            self.enc[BLOCK_TAG].append([b, "block {}: sum({}) = 1".format(cnt, block)])        
            
            c = self.or_model.Add(cvar <= bound)
            self.enc[BLOCK_TAG].append([c, "bound {}: {} <= {}".format(cnt, c, bound)])        

           
            cnt = cnt + 1       
            print(".", end = "", flush=True) 

        print("\n-->Finished enumeration: {} sec".format(resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer1))

        #exit()
        if (debug):
            print("total {}".format(resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer1))
        
        if self.verbose:
            inpvals = self.xgb.readable_sample(sample)
            #print(inpvals)
            self.preamble = []
            #print(self.xgb.feature_names, inpvals, expl)
            #exit()
            for i, (f, v) in enumerate(zip(self.xgb.feature_names, inpvals)):
                if f not in v:
                    self.preamble.append('{0} = {1}'.format(f, v))
                else:
                    self.preamble.append(v)

            print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))


            for h, expl in enumerate(expls):
                inpvals = self.xgb.readable_sample(sample)
                #print(inpvals)
                self.preamble = []
                #print(self.xgb.feature_names, inpvals, expl)
                #exit()

                for i, v in enumerate(self.xgb.feature_names):
                    if i in expl:
                        s = 'NOT {0} =  {1}'.format(self.xgb.feature_names[i], inpvals[i])
                        #print(s)
                        self.preamble.append(s)
                    #else:
                    #    self.preamble.append(v)
                #print(expl, self.preamble)
                print('  explanation:  "IF {0} THEN NOT {1}"'.format(' AND '.join(self.preamble), self.output))
                print("  # {}th hypos   left: {}".format(h, len(expl)))
        
        return expls
        #if self.optns.verb:
        #    print('xgb scores:', cscores)
        #    print('enc scores:', escores)

#     def save_to(self, outfile):
#         """
#             Save the encoding into a file with a given name.
#         """
# 
#         if outfile.endswith('.txt'):
#             outfile = outfile[:-3] + 'smt2'
# 
#         write_smtlib(self.enc, outfile)
# 
#         # appending additional information
#         with open(outfile, 'r') as fp:
#             contents = fp.readlines()
# 
#         # comments
#         comments = ['; features: {0}\n'.format(', '.join(self.feats)),
#                 '; classes: {0}\n'.format(self.nofcl)]
# 
#         if self.intvs:
#             for f in self.xgb.extended_feature_names_as_array_strings:
#                 c = '; i {0}: '.format(f)
#                 c += ', '.join(['{0}<->{1}'.format(u, v) for u, v in zip(self.intvs[f], self.ivars[f])])
#                 comments.append(c + '\n')
# 
#         contents = comments + contents
#         with open(outfile, 'w') as fp:
#             fp.writelines(contents)
# 
#     def load_from(self, infile):
#         """
#             Loads the encoding from an input file.
#         """
# 
#         with open(infile, 'r') as fp:
#             file_content = fp.readlines()
# 
#         # empty intervals for the standard encoding
#         self.intvs, self.imaps, self.ivars = {}, {}, {}
# 
#         for line in file_content:
#             if line[0] != ';':
#                 break
#             elif line.startswith('; i '):
#                 f, arr = line[4:].strip().split(': ', 1)
#                 f = f.replace('-', '_')
#                 self.intvs[f], self.imaps[f], self.ivars[f] = [], {}, []
# 
#                 for i, pair in enumerate(arr.split(', ')):
#                     ub, symb = pair.split('<->')
# 
#                     if ub[0] != '+':
#                         ub = float(ub)
#                     symb = Symbol(symb, typename=BOOL)
# 
#                     self.intvs[f].append(ub)
#                     self.ivars[f].append(symb)
#                     self.imaps[f][ub] = i
# 
#             elif line.startswith('; features:'):
#                 self.feats = line[11:].strip().split(', ')
#             elif line.startswith('; classes:'):
#                 self.nofcl = int(line[10:].strip())
# 
#         parser = SmtLibParser()
#         script = parser.get_script(StringIO(''.join(file_content)))
# 
#         self.enc = script.get_last_formula()
# 
#     def access(self):
#         """
#             Get access to the encoding, features names, and the number of
#             classes.
#         """
# 
#         return self.enc, self.intvs, self.imaps, self.ivars, self.feats, self.nofcl
#                     if (False):
#                         print(solution_o - solution_n)
#   
#                         cnt = 0
#                         for i,feat in  self.categories_enc.items():
#                                     print(i + "  ", end = "")
#                                     print(feat)
#                                     id = i.split('_')[1]                                                
#                                     vid  = int(id[1:])
#                                     print("sol[{}] = {}".format(vid, solution_o[vid]))
#     
#                                     for j,f in enumerate(feat):
#                                         if (f == 0):
#                                             sample_internal.append(int(f))
#                                             sample_internal_feat.append("{}: {}={}".format(cnt, f,0))    
#                                             print(f, " ~0 -> ", sample_internal_o[cnt], cnt, end = "") 
#                                             assert(sample_internal_o[cnt] == 0)
#                                             cnt = cnt + 1  
#     
#                                             continue
#                                         if (self.solver.Value(self.get_feature_bool_var(self.or_model, f)) == 1):                          
#                                             sample_internal.append(1)
#                                             sample_internal_feat.append("{}: {}={}".format(cnt, f,1))
#                                             print(f, " 1 -> ", sample_internal_o[cnt], end = "")                                        
#                                             assert(sample_internal_o[cnt] == 1)
#     
#                                         else:
#                                             sample_internal.append(0)
#                                             sample_internal_feat.append("{}: {}={}".format(cnt, f,0))    
#                                             print(f, " 0 -> ", sample_internal_o[cnt], cnt,  end = "")         
#                                             assert(sample_internal_o[cnt] == 0)
#                                             
#                                         cnt = cnt + 1                               
#                                             #if (f in hypos_feat):
#                                                 #print(f)
#                                                 #print(hypos_feat)
#                                                 #exit()
#                                                 #block.append(self.get_feature_bool_var(self.or_model, f))
#                                                 #block_str.append(self.get_feature_bool_var(self.or_model, f).__str__()) 
#                                     print("")                               
#                                              

#3: python3.6 dxp.py -N 2000  -e ortools -M  -x '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,10,14,14,15,13,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,15,15,15,14,15,14,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,15,4,2,1,4,12,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,0,0,0,0,4,15,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,15,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,6,15,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,9,14,13,15,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,15,15,15,15,15,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,15,15,15,15,15,15,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,15,15,13,9,2,12,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,4,0,0,0,4,15,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,15,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,7,6,1,0,0,0,0,0,0,8,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,15,11,0,0,0,0,0,0,3,15,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,15,15,14,8,3,7,8,10,15,14,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,14,15,15,15,15,15,15,14,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,10,12,9,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0' --xtype contrastive -v temp/digits_3_5_20000_16_data/digits_3_5_20000_16_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
#5: python3.6 dxp.py -N 2000  -e ortools -M  -x '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,9,7,4,6,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,15,10,9,12,10,12,13,10,9,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,12,0,0,0,0,0,1,2,8,14,15,13,13,9,8,6,3,8,5,0,0,0,0,0,0,0,0,14,9,0,0,0,0,0,0,0,0,0,3,4,9,9,9,9,9,15,10,0,0,0,0,0,0,0,0,15,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,10,1,0,0,0,0,0,0,0,0,12,4,0,0,3,4,9,13,13,10,6,0,0,0,0,0,0,4,1,0,0,0,0,0,0,0,0,0,10,8,8,13,14,12,7,5,1,6,8,13,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,15,10,5,1,0,0,0,0,0,0,6,15,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,5,0,0,0,0,0,0,0,0,0,0,2,12,13,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,12,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,15,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,13,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,15,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,1,0,0,0,0,0,0,0,0,0,13,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,4,0,0,0,0,0,0,0,0,2,15,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,12,0,0,0,0,0,0,0,0,7,15,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,12,10,0,2,2,0,1,2,6,13,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,14,13,15,15,12,15,15,14,11,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1' --xtype contrastive -v temp/digits_3_5_20000_16_data/digits_3_5_20000_16_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
# real python3.6 dxp.py -N 2000  -e ortools -x '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,13,15,15,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,10,15,15,15,15,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,14,15,15,15,11,7,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,14,15,15,13,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,14,15,15,11,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,14,15,15,11,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,15,15,11,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,15,15,13,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,15,15,15,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,13,15,15,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,15,15,14,7,10,5,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,15,15,15,15,15,15,15,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,15,15,15,15,15,15,15,15,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,15,15,15,11,6,5,10,15,15,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,15,15,15,6,0,0,0,0,8,15,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,15,12,0,0,0,0,0,8,15,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,15,13,1,0,2,5,11,15,15,15,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,15,15,15,15,15,15,15,15,15,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,15,15,15,15,15,15,14,11,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,15,15,15,11,8,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0' --xtype contrastive -v temp/real_fake_20000_16_data/real_fake_20000_16_data_nbestim_100_maxdepth_6_testsplit_0.2.mod.pkl
# fake python3.6 dxp.py -N 2000  -e ortools  -x '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,6,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,8,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,15,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,12,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,15,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,13,6,0,0,0,0,0,0,0,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,15,8,0,0,0,0,7,15,15,15,15,13,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,15,0,0,7,15,14,7,6,2,3,15,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,9,0,9,12,6,0,0,0,0,0,9,15,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,15,14,9,6,6,0,0,0,0,0,0,15,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,15,9,5,0,0,0,0,0,2,9,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,15,15,1,0,0,0,9,14,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,15,15,14,13,15,9,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,5,8,5,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,0,1' --xtype contrastive -v temp/real_fake_20000_16_data/real_fake_20000_16_data_nbestim_100_maxdepth_6_testsplit_0.2.mod.pkl
