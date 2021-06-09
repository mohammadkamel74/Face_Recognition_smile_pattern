#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sklearn
import numpy as np
from sklearn import svm,metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,roc_curve, auc,multilabel_confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, confusion_matrix


# In[ ]:


data = pd.read_csv("train.csv",header=None)
datatest = pd.read_csv("test.csv",header=None)


# In[ ]:


x=np.zeros([int(len(data)/13),5])
y=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x[i]=np.array(data.iloc[13*i][:5])
    y[i]=np.array(data.iloc[13*i][5])


xx=np.zeros([int(len(datatest)/13),5])
yy=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx[i]=np.array(datatest.iloc[13*i][:5])
    yy[i]=np.array(datatest.iloc[13*i][5])

###########################################################################################################
# we used GridSearchCV which exhaustively considers all parameter combinations
# We determined The best parameters that can be determined by grid search techniques to search the hyper-parameter space
# for the best cross validation score. the below tuned parameters is used in our GridSearchCV:
#############################Set the parameters by cross-validation#########################################

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

#######################################SVM FOR THE FIRST PART##############################################
clf = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf.fit(x, y)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true, y_pred = yy, clf.predict(xx)


# In[ ]:



###########################################second feature#####################################################

x1=np.zeros([int(len(data)/13),5])
y1=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x1[i]=np.array(data.iloc[13*i+1][:5])
    y1[i]=np.array(data.iloc[13*i+1][5])


xx1=np.zeros([int(len(datatest)/13),5])
yy1=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx1[i]=np.array(datatest.iloc[13*i+1][:5])
    yy1[i]=np.array(datatest.iloc[13*i+1][5])


############################################SVM FOR THE Second PART##########################################
clf1 = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf1.fit(x1, y1)

print("Best parameters set found on development set:")
print()
print(clf1.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf1.cv_results_['mean_test_score']
stds = clf1.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf1.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

y_true1, y_pred1 = yy1, clf1.predict(xx1)


# In[ ]:



###########################################third feature#####################################################

x2=np.zeros([int(len(data)/13),5])
y2=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x2[i]=np.array(data.iloc[13*i+2][:5])
    y2[i]=np.array(data.iloc[13*i+2][5])


xx2=np.zeros([int(len(datatest)/13),5])
yy2=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx2[i]=np.array(datatest.iloc[13*i+2][:5])
    yy2[i]=np.array(datatest.iloc[13*i+2][5])


############################################SVM FOR THE 3rd PART##########################################
clf2 = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf2.fit(x2, y2)

print("Best parameters set found on development set:")
print()
print(clf2.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf2.cv_results_['mean_test_score']
stds = clf2.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf2.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true2, y_pred2 = yy2, clf2.predict(xx2)


# In[ ]:




###########################################forth feature#####################################################

x3=np.zeros([int(len(data)/13),5])
y3=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x3[i]=np.array(data.iloc[13*i+3][:5])
    y3[i]=np.array(data.iloc[13*i+3][5])


xx3=np.zeros([int(len(datatest)/13),5])
yy3=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx3[i]=np.array(datatest.iloc[13*i+3][:5])
    yy3[i]=np.array(datatest.iloc[13*i+3][5])


############################################SVM FOR THE Second PART##########################################
clf3 = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf3.fit(x3, y3)

print("Best parameters set found on development set:")
print()
print(clf3.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf3.cv_results_['mean_test_score']
stds = clf3.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf3.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

y_true3, y_pred3 = yy3, clf.predict(xx3)


# In[ ]:



 ###########################################fifth feature#####################################################

x4=np.zeros([int(len(data)/13),5])
y4=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x4[i]=np.array(data.iloc[13*i+4][:5])
    y4[i]=np.array(data.iloc[13*i+4][5])


xx4=np.zeros([int(len(datatest)/13),5])
yy4=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx4[i]=np.array(datatest.iloc[13*i+4][:5])
    yy4[i]=np.array(datatest.iloc[13*i+4][5])


############################################SVM FOR THE 5th PART##########################################
clf4 = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf4.fit(x4, y4)

print("Best parameters set found on development set:")
print()
print(clf4.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf4.cv_results_['mean_test_score']
stds = clf4.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf4.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true4, y_pred4 = yy4, clf4.predict(xx4)


# In[ ]:



###########################################6th feature#####################################################

x5=np.zeros([int(len(data)/13),5])
y5=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x5[i]=np.array(data.iloc[13*i+5][:5])
    y5[i]=np.array(data.iloc[13*i+5][5])


xx5=np.zeros([int(len(datatest)/13),5])
yy5=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx5[i]=np.array(datatest.iloc[13*i+5][:5])
    yy5[i]=np.array(datatest.iloc[13*i+5][5])


############################################SVM FOR THE 6th PART##########################################
clf5 = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf5.fit(x5, y5)

print("Best parameters set found on development set:")
print()
print(clf5.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf5.cv_results_['mean_test_score']
stds = clf5.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf5.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true5, y_pred5 = yy5, clf5.predict(xx5)

###########################################7th feature#####################################################


# In[ ]:



x6=np.zeros([int(len(data)/13),5])
y6=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x6[i]=np.array(data.iloc[13*i+6][:5])
    y6[i]=np.array(data.iloc[13*i+6][5])


xx6=np.zeros([int(len(datatest)/13),5])
yy6=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx6[i]=np.array(datatest.iloc[13*i+6][:5])
    yy6[i]=np.array(datatest.iloc[13*i+6][5])

############################################SVM FOR THE 7th PART##########################################
clf6 = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf6.fit(x6, y6)


means = clf6.cv_results_['mean_test_score']
stds = clf6.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf6.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true6, y_pred6 = yy6, clf6.predict(xx6)


# In[ ]:



x7=np.zeros([int(len(data)/13),5])
y7=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x7[i]=np.array(data.iloc[13*i+7][:5])
    y7[i]=np.array(data.iloc[13*i+7][5])


xx7=np.zeros([int(len(datatest)/13),5])
yy7=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx7[i]=np.array(datatest.iloc[13*i+7][:5])
    yy7[i]=np.array(datatest.iloc[13*i+7][5])


############################################SVM FOR THE forth PART##########################################


clf7 = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf7.fit(x7, y7)

print("Best parameters set found on development set:")
print()
print(clf7.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf7.cv_results_['mean_test_score']
stds = clf7.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf7.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()


y_true7, y_pred7 = yy7, clf7.predict(xx7)


# In[ ]:




# In[10]
 ###########################################9th feature#####################################################

x8=np.zeros([int(len(data)/13),5])
y8=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x8[i]=np.array(data.iloc[13*i+8][:5])
    y8[i]=np.array(data.iloc[13*i+8][5])


xx8=np.zeros([int(len(datatest)/13),5])
yy8=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx8[i]=np.array(datatest.iloc[13*i+8][:5])
    yy8[i]=np.array(datatest.iloc[13*i+8][5])


############################################SVM FOR THE Second PART##########################################
clf8 = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf8.fit(x8, y8)

print("Best parameters set found on development set:")
print()
print(clf8.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf8.cv_results_['mean_test_score']
stds = clf8.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf8.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

y_true8, y_pred8 = yy8, clf8.predict(xx8)


# In[ ]:




###########################################8th feature#####################################################

###########################################10th feature#####################################################

x9=np.zeros([int(len(data)/13),5])
y9=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x9[i]=np.array(data.iloc[13*i+9][:5])
    y9[i]=np.array(data.iloc[13*i+9][5])


xx9=np.zeros([int(len(datatest)/13),5])
yy9=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx9[i]=np.array(datatest.iloc[13*i+9][:5])
    yy9[i]=np.array(datatest.iloc[13*i+9][5])


############################################SVM FOR THE 10th PART##########################################
clf9 = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf9.fit(x9, y9)

print("Best parameters set found on development set:")
print()
print(clf9.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf9.cv_results_['mean_test_score']
stds = clf9.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf9.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

y_true9, y_pred9 = yy9, clf9.predict(xx9)


# In[ ]:




###########################################11th feature#####################################################

x10=np.zeros([int(len(data)/13),5])
y10=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x10[i]=np.array(data.iloc[13*i+10][:5])
    y10[i]=np.array(data.iloc[13*i+10][5])


xx10=np.zeros([int(len(datatest)/13),5])
yy10=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx10[i]=np.array(datatest.iloc[13*i+10][:5])
    yy10[i]=np.array(datatest.iloc[13*i+10][5])


############################################SVM FOR THE 11th PART##########################################
clf10 = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf10.fit(x10, y10)

print("Best parameters set found on development set:")
print()
print(clf10.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf10.cv_results_['mean_test_score']
stds = clf10.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf10.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true10, y_pred10 = yy10, clf10.predict(xx10)


# In[ ]:




###########################################12th feature#####################################################

x11=np.zeros([int(len(data)/13),5])
y11=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x11[i]=np.array(data.iloc[13*i+11][:5])
    y11[i]=np.array(data.iloc[13*i+11][5])


xx11=np.zeros([int(len(datatest)/13),5])
yy11=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx11[i]=np.array(datatest.iloc[13*i+11][:5])
    yy11[i]=np.array(datatest.iloc[13*i+11][5])


############################################SVM FOR THE 12th PART##########################################
clf11 = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf11.fit(x11, y11)

print("Best parameters set found on development set:")
print()
print(clf11.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf11.cv_results_['mean_test_score']
stds = clf11.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf11.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


y_true11, y_pred11 = yy11, clf11.predict(xx11)


# In[ ]:




 ###########################################third feature#####################################################

x12=np.zeros([int(len(data)/13),5])
y12=np.zeros([int(len(data)/13),1])
for i in range(int(len(data)/13)):
    x12[i]=np.array(data.iloc[13*i+12][:5])
    y12[i]=np.array(data.iloc[13*i+12][5])


xx12=np.zeros([int(len(datatest)/13),5])
yy12=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx12[i]=np.array(datatest.iloc[13*i+12][:5])
    yy12[i]=np.array(datatest.iloc[13*i+12][5])


############################################SVM FOR THE Second PART##########################################
clf12 = GridSearchCV(
    SVC(), tuned_parameters, scoring='accuracy',cv=3
)
clf12.fit(x12, y12)

print("Best parameters set found on development set:")
print()
print(clf12.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf12.cv_results_['mean_test_score']
stds = clf12.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf12.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
y_true12, y_pred12 = yy12, clf12.predict(xx12)


# In[ ]:




# In[11]:


datatest = pd.read_csv("test.csv",header=None)
xx=np.zeros([int(len(datatest)/13),5])
yy=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx[i]=np.array(datatest.iloc[13*i][:5])
    yy[i]=np.array(datatest.iloc[13*i][5])
#################################################################
xx1=np.zeros([int(len(datatest)/13),5])
yy1=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx1[i]=np.array(datatest.iloc[13*i+1][:5])
    yy1[i]=np.array(datatest.iloc[13*i+1][5])

#################################################################
xx2=np.zeros([int(len(datatest)/13),5])
yy2=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx2[i]=np.array(datatest.iloc[13*i+2][:5])
    yy2[i]=np.array(datatest.iloc[13*i+2][5])
#################################################################
xx3=np.zeros([int(len(datatest)/13),5])
yy3=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx3[i]=np.array(datatest.iloc[13*i+3][:5])
    yy3[i]=np.array(datatest.iloc[13*i+3][5])

#################################################################
xx4=np.zeros([int(len(datatest)/13),5])
yy4=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx4[i]=np.array(datatest.iloc[13*i+4][:5])
    yy4[i]=np.array(datatest.iloc[13*i+4][5])
#################################################################
xx5=np.zeros([int(len(datatest)/13),5])
yy5=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx5[i]=np.array(datatest.iloc[13*i+5][:5])
    yy5[i]=np.array(datatest.iloc[13*i+5][5])
#################################################################
xx6=np.zeros([int(len(datatest)/13),5])
yy6=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx6[i]=np.array(datatest.iloc[13*i+6][:5])
    yy6[i]=np.array(datatest.iloc[13*i+6][5])

#################################################################

xx7=np.zeros([int(len(datatest)/13),5])
yy7=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx7[i]=np.array(datatest.iloc[13*i+7][:5])
    yy7[i]=np.array(datatest.iloc[13*i+7][5])
#################################################################
xx8=np.zeros([int(len(datatest)/13),5])
yy8=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx8[i]=np.array(datatest.iloc[13*i+8][:5])
    yy8[i]=np.array(datatest.iloc[13*i+8][5])
#################################################################
xx9=np.zeros([int(len(datatest)/13),5])
yy9=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx9[i]=np.array(datatest.iloc[13*i+9][:5])
    yy9[i]=np.array(datatest.iloc[13*i+9][5])
#################################################################
xx10=np.zeros([int(len(datatest)/13),5])
yy10=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx10[i]=np.array(datatest.iloc[13*i+10][:5])
    yy10[i]=np.array(datatest.iloc[13*i+10][5])
#################################################################
xx11=np.zeros([int(len(datatest)/13),5])
yy11=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx11[i]=np.array(datatest.iloc[13*i+11][:5])
    yy11[i]=np.array(datatest.iloc[13*i+11][5])
#################################################################

xx12=np.zeros([int(len(datatest)/13),5])
yy12=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    xx12[i]=np.array(datatest.iloc[13*i+12][:5])
    yy12[i]=np.array(datatest.iloc[13*i+12][5])
#################################################################
#################################################################

y_pred = clf.predict(xx)
y_pred1 = clf1.predict(xx1)
y_pred2 = clf2.predict(xx2)
y_pred3 = clf3.predict(xx3)
y_pred4 = clf4.predict(xx4)
y_pred5 = clf5.predict(xx5)
y_pred6 = clf6.predict(xx6)
y_pred7 = clf7.predict(xx7)
y_pred8 = clf8.predict(xx8)
y_pred9 = clf9.predict(xx9)
y_pred10 = clf10.predict(xx10)
y_pred11 = clf11.predict(xx11)
y_pred12 = clf12.predict(xx12)


# In[12]:


# In[ ]:



 ###########################################second feature#####################################################

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

# when the data set is small one of the best way to have a multi label classifier is 
# Naive Bayes if the features are independent. 
# Also, Gaussian naive Bayes classifiers works the best with numerical data.

clfN = GaussianNB()
clfN.fit(x, y)
y_predN=clfN.predict(xx)


clfN1 = GaussianNB()
clfN1.fit(x1, y1)
y_predN1=clfN1.predict(xx1)


clfN2 = GaussianNB()
clfN2.fit(x2, y2)
y_predN2=clfN2.predict(xx2)


clfN3= GaussianNB()
clfN3.fit(x3, y3)
y_predN3=clfN3.predict(xx3)



clfN4= GaussianNB()
clfN4.fit(x4, y4)
y_predN4=clfN4.predict(xx4)

clfN5= GaussianNB()
clfN5.fit(x5, y5)
y_predN5=clfN5.predict(xx5)

clfN6 = GaussianNB()
clfN6.fit(x6, y6)
y_predN6=clfN6.predict(xx6)

clfN7 = GaussianNB()
clfN7.fit(x7, y7)
y_predN7=clfN7.predict(xx7)


clfN8 = GaussianNB()
clfN8.fit(x8, y8)
y_predN8=clfN8.predict(xx8)

clfN9= GaussianNB()
clfN9.fit(x9, y9)
y_predN9=clfN9.predict(xx9)

clfN10= GaussianNB()
clfN10.fit(x10, y10)
y_predN10=clfN10.predict(xx10)

clfN11= GaussianNB()
clfN11.fit(x11, y11)
y_predN11=clfN11.predict(xx11)

clfN12= GaussianNB()
clfN12.fit(x12, y12)
y_predN12=clfN12.predict(xx12)

l=[]

################################################FIND THE FIRST 2 MOST COMMON GRIOUP PREDICTED BY SVM AND NAIVE BASYSIAN################################################

from collections import Counter
for i in range(len(y_pred)):
    a = [y_pred[i],y_pred1[i],y_pred2[i],y_pred3[i],y_pred4[i],y_pred5[i],y_pred6[i],y_pred7[i],y_pred8[i],y_pred9[i],y_pred10[i],y_pred11[i],y_pred12[i]] 
    c = Counter(a)
    l.append(c.most_common(2))
    
N=[]

from collections import Counter
for i in range(len(y_pred)):
    a = [y_predN[i],y_predN1[i],y_predN2[i],y_predN3[i],y_predN4[i],y_predN5[i],y_predN6[i],y_predN7[i],y_predN8[i],y_predN9[i],y_predN10[i],y_predN11[i],y_predN12[i]] 
    c = Counter(a)
    N.append(c.most_common(2))

######################max voting first part###########################33
######################if there exist two same predicted class by two different algorithm add their number####3
for i in range(len(N)):
    for x,v in N[i]:
        for z,k in l[i]:
            if x==z:
                N[i][0]=(x,k+v)
                
#######################choose the class with maximum selected by algorithms as the final decision#############
import numpy as np
final_decision=np.zeros([len(N),i])
for i in range(len(N)):
    max=0
    for x,v in N[i]:
        if v>=max: 
            max=v
            final_decision[i]=x
        for z,k in l[i]:
            if k>=max:
                final_decision[i]=z



# In[22]:


# In[ ]:


from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, confusion_matrix

datatest = pd.read_csv("test.csv",header=None)
YY=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    YY[i]=np.array(datatest.iloc[13*i][5])


print("the ground truth classification:",YY)
y_score=final_decision[:,0]
print("the output classification:",y_score)

CM=metrics.accuracy_score(YY, y_score)
print('accuracy is:', CM)

   
YY = label_binarize(YY, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12])
n_classes = YY.shape[1]
y_score = label_binarize(y_score, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12])

#############################################ROC##################################################
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(YY[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(YY.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw = 2
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('roccurve.png')
plt.show()
##############################################################################################

###################################precision-recall############################################
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(YY[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(YY[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(YY.ravel(),y_score.ravel())
average_precision["micro"] = average_precision_score(YY, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal','blue','green','red','cyan','magenta','yellow','black','#eeefff'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
plt.savefig("percisionrecallcureve.png")
plt.show()

###########################################################################################################



MSE=mean_squared_error(YY, y_score)
print("the mean square error is",MSE)
MAE=mean_absolute_error(YY, y_score)
print("the mean absoulte error is",MAE)
LL=log_loss(YY, y_score)
print(LL)
CM=multilabel_confusion_matrix(YY, y_score)
for i in range(CM.shape[0]):
    plt.imshow(CM[i], cmap='binary')
    plt.savefig(str(i)+'.png')


datatest = pd.read_csv("test.csv",header=None)
YY=np.zeros([int(len(datatest)/13),1])
for i in range(int(len(datatest)/13)):
    YY[i]=np.array(datatest.iloc[13*i][5])
    
    
CM=metrics.accuracy_score(YY, y_predN)
print('accuracy is for NB:', CM)
CM=metrics.accuracy_score(YY, y_predN1)
print('accuracy is for NB1:', CM)
CM=metrics.accuracy_score(YY, y_predN2)
print('accuracy is for NB2:', CM)
CM=metrics.accuracy_score(YY, y_predN3)
print('accuracy is for NB3:', CM)
CM=metrics.accuracy_score(YY, y_predN4)
print('accuracy is for NB4:', CM)
CM=metrics.accuracy_score(YY, y_predN5)
print('accuracy is for NB5:', CM)
CM=metrics.accuracy_score(YY, y_predN6)
print('accuracy is for NB6:', CM)
CM=metrics.accuracy_score(YY, y_predN7)
print('accuracy is for NB7:', CM)
CM=metrics.accuracy_score(YY, y_predN8)
print('accuracy is for NB8:', CM)
CM=metrics.accuracy_score(YY, y_predN9)
print('accuracy is for NB9:', CM)

CM=metrics.accuracy_score(YY, y_predN10)
print('accuracy is for NB10:', CM)
CM=metrics.accuracy_score(YY, y_predN11)
print('accuracy is for NB11:', CM)
CM=metrics.accuracy_score(YY, y_predN12)
print('accuracy is for NB12:', CM)


# In[ ]:


final_decision[:,0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




