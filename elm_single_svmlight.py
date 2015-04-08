#!/usr/bin/python
# -*- coding: utf-8 -*-

print __doc__


# Code source: Gael Varoqueux
#              Andreas Mueller
# Modified for Documentation merge by Jaques Grobler
# Modified for Extreme Learning Machine Classifiers by David Lambert
# License: BSD

import numpy as np
# import pylab as pl

# from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

from elm import GenELMClassifier
from random_layer import RBFRandomLayer, MLPRandomLayer
from numpy import *
import shlex, subprocess, sys, time
from sklearn.datasets import load_svmlight_file

def get_data_bounds(X):
    h = .02  # step size in the mesh

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    return (x_min, x_max, y_min, y_max, xx, yy)

def make_datasets():
    return [make_moons(n_samples=200, noise=0.3, random_state=0),
            make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1),
            make_linearly_separable()]


def make_classifiers():

    names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)",
             "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]

    nh = 10

    # pass user defined transfer func
    sinsq = (lambda x: np.power(np.sin(x), 2.0))
    srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)

    # use internal transfer funcs
    srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')

    srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')

    srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')

    # use gaussian RBF
    srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)

    log_reg = LogisticRegression()

    classifiers = [GenELMClassifier(hidden_layer=srhl_tanh),
                   GenELMClassifier(hidden_layer=srhl_tanh, regressor=log_reg),
                   GenELMClassifier(hidden_layer=srhl_sinsq),
                   GenELMClassifier(hidden_layer=srhl_tribas),
                   GenELMClassifier(hidden_layer=srhl_hardlim),
                   GenELMClassifier(hidden_layer=srhl_rbf)]

    return names, classifiers


def make_linearly_separable():
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, random_state=1,
                               n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return (X, y)

###############################################################################

# INPUTDATA=sys.argv[1]
INPUTDATA = '/home/cao/Projects/gblm/data/small.txt'
# read in targets

def calcAccurary(predictions, labels):
    Acc = 0
    for i in range(size(labels)):
        if labels[i]==predictions[i]:
          Acc = Acc + 1
    Acc = Acc / float(size(labels))
    return Acc

def readtargets(filename):
        p1=subprocess.Popen(shlex.split('cut -f 1,2 -d\  %s' % filename),stdout=subprocess.PIPE);
        output=[a.split(' ',1) for a in p1.stdout.read().split('\n')  if len(a)>0];
        y_train = []
        ys = [float(a[0]) for a in output];
        qs = [int(a[1].split(':')[1]) for a in output];
        # y_train = y_train.append(ys)
        # ys = ys.append(qs)
        # print shape(ys)
        # y_train = y_train.append(qs)
        return (ys, qs)

ys, qs = readtargets(INPUTDATA)
# print targets[:10]
y_pred = zeros(shape(ys)); y_preds = zeros(shape(ys))
X_train, y_train = load_svmlight_file(INPUTDATA) # y_train is obsolete
X_test = X_train; y_test = ys
names, classifiers = make_classifiers()
clf = classifiers[0]
for iter in range(9):
	y_target = mat(ys) - mat(y_pred);
	# print y_target
	# iterate over classifiers
	clf.fit(X_train, list(array(y_target).reshape(-1)))
	y_pred = clf.predict(X_train)
	y_preds = y_preds + .1 * y_pred
	# print mat(y_pred)
	# print score
acc = calcAccurary(y_preds, y_test)
print y_preds
print "Final score:\n"
print acc
"""
for name, clf in zip(names, classifiers):
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	print score
	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, m_max]x[y_min, y_max].
	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
"""
