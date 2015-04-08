from numpy import *
import shlex, subprocess, sys, time
from sklearn.datasets import load_svmlight_file

INPUTDATA=sys.argv[1]
# read in targets

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
        return ys, qs 

ys, qs = readtargets(INPUTDATA)

# print targets[:10]
X_train = load_svmlight_file(INPUTDATA)
print shape(y_train) 
# print X_train[:5,:20]
# print y_train[:10]
# print queries[:10]
