#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Test code of Extreme Learning machine
# Wrote by Guanqun Cao
# Last update on 08/04/2015
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#!/usr/bin/python
# -*- coding: utf-8 -*-

# print __doc__
print "Kick off extreme learning machine.. "


import numpy as np
# import pylab as pl

# from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from elm import ELMClassifier, ELMRegressor, GenELMClassifier, GenELMRegressor
from random_layer import RBFRandomLayer, MLPRandomLayer
from numpy import *
import shlex, subprocess, sys, time
from sklearn.datasets import load_svmlight_file

def loadDataSet(fileName, low_limit, up_limit):      #general function to parse tab -delimited floats
    #numFeat = len(open(fileName).readline().split()) #get number of fields
    dataMat = []; labelMat = []
    cnt = 0
    #for line in fr.readlines():
    with open(fileName) as fr:
        for line in fr:
            if cnt >= float(low_limit) and cnt <= float(up_limit):
                lineArr =zeros(2000)
                curLine = line.strip().split()
                numFeat = len(curLine)
                for i in range(numFeat-1):
            #print curLine[i+1].strip().split(':')
                    num, feat =curLine[i+1].strip().split(':')
                    lineArr[float(num)-1]=float(feat)
            #print lineArr
                dataMat.append(lineArr)
                labelMat.append(int(curLine[0]))
            elif cnt > float(up_limit):
              break
            cnt += 1
    return dataMat,labelMat



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

if __name__ == '__main__':
	# INPUTDATA=sys.argv[1]
	INPUTDATA = '/home/cao/Dataset2/epsilon/epsilon_normalized'
	TESTDATA = '/home/cao/Dataset2/epsilon/epsilon_normalized.t'
	LRATE = .1
	low_limit = 0
        up_limit = 1000
        # Load training file..
        datArr,labelArr = loadDataSet(INPUTDATA, low_limit, up_limit)
	labelMat = mat(labelArr)
	# labelMat[labelMat == -1] = 0 
	# labelArr = array(labelMat)
        testDatArr,testLabelArr = loadDataSet(TESTDATA, low_limit, up_limit)
	testLabelMat = mat(testLabelArr)
	# testLabelMat[testLabelMat == -1] = 0 
	# testLabelArr = array(testLabelMat)
	# print targets[:10]
	y_train_pred = zeros(shape(labelArr)); y_preds = zeros(shape(testLabelArr))
	X_test = testDatArr; y_test = testLabelArr
	# clf = ELMClassifier(n_hidden=100, activation_func='gaussian', alpha=0.0, random_state=1)
	# clf = ELMClassifier(n_hidden=100 000, activation_func='hardlim', alpha=1.0, random_state=0)
	clf = ELMClassifier(n_hidden=100, activation_func='hardlim', alpha=1.0, random_state=1)
	for iter in range(2):
		# y_target = mat(labelArr)
		y_target = sign(mat(labelArr) - mat(y_train_pred));
		print y_target
		print y_train_pred
		# print shape(y_target)
		# iterate over classifiers
		clf.fit(datArr, list(array(y_target).reshape(-1)))
		y_train_pred = clf.predict(datArr)
		y_pred = clf.predict(testDatArr)
		acc = calcAccurary(sign(y_train_pred), labelArr)
		print "Training accurarcy:\n"
		print acc
		acc = calcAccurary(sign(y_pred), y_test)
		print acc
		y_preds = y_preds + LRATE * y_pred
	print mat(y_preds)
	acc = calcAccurary(sign(y_preds), y_test)
	# print y_preds
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
