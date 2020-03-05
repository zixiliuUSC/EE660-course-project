#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.externals import joblib
#import warnings
#warnings.filterwarnings("ignore")
df_train = pd.read_csv('../input/train.csv')
df_y_train = pd.read_csv('../input/y_train.csv')
df_val = pd.read_csv('../input/val.csv')
df_y_val = pd.read_csv('../input/y_val.csv')
train = df_train['review'].to_numpy()
y_train = df_y_train['rate'].to_numpy()
val = df_val['review'].to_numpy()
y_val = df_y_val['rate'].to_numpy()

# the data set is merge here for crossvalidation in the Bayesian Inference analysis. 
'''
train_total = pd.DataFrame({'review':train.tolist()+val.tolist()})
train_y_total = pd.DataFrame({'rate':y_train.tolist()+y_val.tolist()})
train_total.to_csv('../input/train_total.csv')
train_y_total.to_csv('../input/train_y_total.csv')
'''

vectorizer = TfidfVectorizer(stop_words='english',use_idf=True)
model_tr = vectorizer.fit_transform(train)
model_val = vectorizer.transform(val)


# In[14]:


from sklearn.decomposition import PCA
#from sklearn.preprocessing import normalize
#model_tr_normalize = normalize(model_tr, norm='l2', axis=0, copy=True, return_norm=False)
pca = PCA(n_components = 0.95)
pca.fit(model_tr.toarray())
reduced = pca.transform(model_tr.toarray())
print(reduced.shape)


# In[16]:


from sklearn.decomposition import PCA
model_tr1 = model_tr.copy()
model_val1 = model_val.copy()
#pca = joblib.load('../input/model/pca.pkl')
model_val = pca.transform(model_val.toarray())
model_tr = pca.transform(model_tr.toarray())


# In[4]:


# Logistic regression, L2 regularization, weighted loss, hyperparameter selection use validation set, macro-weighted score to select model
# reason: we assume that all the variables are necessary for classification so we use a L2 regularizer. 
# and we give more weight to the minor class to remedy for the class imblance. 
import warnings
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
n_alphas = 76
#alphas = np.exp(np.linspace(-5,10,n_alphas))
C = np.exp(np.linspace(-5,10,n_alphas))
train_score = []
val_score = []
for i in C:
    m = LogisticRegression(C=i,class_weight='balanced').fit(model_tr, y_train)
    s = f1_score(y_val, m.predict(model_val), average='macro')
    train_score.append(f1_score(y_train, m.predict(model_tr), average='macro'))
    val_score.append(f1_score(y_val, m.predict(model_val), average='macro'))
    print ('Val macro score is', s)
#plt.plot(np.log10(C),train_score,label='train')
#plt.plot(np.log10(C),val_score,label='val')
#plt.title('macro score of training set and validation set')
#plt.legend()
#plt.xlabel('log10(C)')
#plt.ylabel('macro score')
#plt.show()
#-------------------


# In[5]:


from sklearn.metrics import confusion_matrix
ax = plt.gca()
# randomly choose 50 coefficients in the 9676 variables 
# to plot figure of coefs vs regularizer. 
ax.plot(C[:len(train_score)], train_score,label='train')
ax.plot(C[:len(train_score)], val_score,label='val')
ax.set_xscale('log')
#ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('$L2 regularizer: log_e(C)$')
plt.ylabel('macro score')
plt.legend()
plt.title('macro score of training set and validation set')
plt.axis('tight')
plt.show()
C_opt = C[np.argmax(val_score)]
m_opt = LogisticRegression(C=C_opt,class_weight='balanced').fit(model_tr, y_train)
s = f1_score(y_val, m_opt.predict(model_val), average='macro')
mat = confusion_matrix(y_val, m_opt.predict(model_val))
print ('Best Val macro score is', s)
print('confusion matrix is: \n',mat)
print('recall of each class:\n',[round(mat[0,0]/mat[0,:].sum(),4), round(mat[1,1]/mat[1,:].sum(),4),
                                  round(mat[2,2]/mat[2,:].sum(),4),round(mat[3,3]/mat[3,:].sum(),4),
                                  round(mat[4,4]/mat[4,:].sum(),4)])
joblib.dump(m_opt,'../input/model/logistic_L2.pkl')


# In[66]:


# ridge regression (L2 regularizer)
# 
from sklearn import linear_model
n_alphas = 50
alphas = np.exp(np.linspace(-10,10,n_alphas))
coefs = []
train_score = []
val_score = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=True)
    ridge.fit(model_tr, y_train)
    train_pre = ridge.predict(model_tr).astype('int')
    train_pre[train_pre > 5] = np.int(5)
    train_pre[train_pre < 0] = np.int(1)
    val_pre = ridge.predict(model_val).astype('int')
    val_pre[val_pre > 5] = np.int(5)
    val_pre[val_pre < 1] = np.int(1)
    train_score.append(f1_score(y_train, train_pre, average='macro'))
    val_score.append(f1_score(y_val, val_pre, average='macro'))
    coefs.append(ridge.coef_)
    
cc = np.array(coefs)

ax = plt.gca()
# randomly choose 50 coefficients in the 9676 variables 
# to plot figure of coefs vs regularizer. 
ax.plot(alphas, cc[:,np.random.randint(0,3192,50)])
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()


# In[67]:


train_score = np.array(train_score)
val_score = np.array(val_score)
ax1 = plt.gca()
# randomly choose 50 coefficients in the 9676 variables 
# to plot figure of coefs vs regularizer. 
ax1.plot(alphas, train_score)
ax1.plot(alphas, val_score)
ax1.set_xscale('log')
ax1.set_xlim(ax1.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('macro score')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

alphas_best = alphas[np.argmax(val_score)]
print('the best macro score on validation set:',np.max(val_score))
ridge = linear_model.Ridge(alpha=alphas_best, fit_intercept=True).fit(model_tr, y_train)
val_pre = ridge.predict(model_val).astype('int')
val_pre[val_pre > 5] = np.int(5)
val_pre[val_pre < 1] = np.int(1)
mat = confusion_matrix(y_val, val_pre)
print('confusion matrix is: \n',mat)
print('f1 score of each class:\n',f1_score(y_val, val_pre, average='macro'))
#print('the number of wi equal to 0 =',len(np.where(np.abs(m.coef_.flatten())==0)[0]))


# In[68]:


joblib.dump(ridge,'../input/model/tf-idf_ridge.pkl')


# ##### Decision Tree
# There are 7 hyperparameters in Decision Tree classifier. Since the dataset in this project has a large size and a large number of variables. Here I use some prior knowledge of this task to determine some hyperparameters and the order hyperparameter tuning. In the tuning process, I will use greedy search to find out the optimal hyperparameter.
# 
# 1."criterion": It defines the criterion whether a node should be split and the two options are gini-index and entropy. When using entropy criterion, or saying the information gain between parent node and child node, the degree of impurity in a node can be better enlarged than using gini-index. But since our task has too many variables, using entropy criterion will result in more overfitting. So gini-index will be chose to be the criterion. 
# 
# 2."splitter": we will choose "best" here so that our algorithm will choose the most important feature to split each time. Although we can use "random" and then merge the leave using some criterion. But because our task is too complecated, splitting the best node each can provide better stability. 
# 
# 3.parameters that I use val score to choose: 
# 
#     The parameters are ordered in their degree of importance to the model and they are choose according to validation set score. 
#     
#     3.1.max_depth: max_depth is the most important hyperparameter in decision, because it mainly decides the ability of the tree to fit the train set. Here I tune max_depth from 3 and fixed minimum of samples in leafs to 5, minmum of samples to split to 10. And give no constraint to maximum of feature number and minimum impurity decrease. 
#     
#     3.2.min_samples_leaf: min_samples_leaf can be use to smooth the boundary and with similar validation score a model with bigger min_samples_leaf can give a better boundary. I use the optimal max_length and the search for the best min_sample_leaf. 
#     
#     3.3.min_samples_split: This hyperparameter also can be use to make the boundary smoother. 
#     
# 

# In[3]:


from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")
clf = DecisionTreeClassifier(random_state=0,criterion='gini',
                            splitter='best' )

depth = np.linspace(3,183,61).astype('int')
train_score1 = []
val_score1 = []
for i in depth:
    m = DecisionTreeClassifier(random_state=0,
                               criterion='gini',
                               splitter='best',
                               max_depth=i,
                               min_samples_leaf=5,
                               min_samples_split=10, 
                               class_weight='balanced'
                              )
    m.fit(model_tr, y_train)
    s = f1_score(y_val, m.predict(model_val), average='macro')
    train_score1.append(f1_score(y_train, m.predict(model_tr), average='macro'))
    val_score1.append(f1_score(y_val, m.predict(model_val), average='macro'))
    #print ('Val macro-weighted score is', s)

train_score1 = np.array(train_score1)
val_score1 = np.array(val_score1)

min_leaf = np.linspace(1,101,51).astype('int')
train_score2 = []
val_score2 = []
for i in min_leaf:
    m = DecisionTreeClassifier(random_state=0,
                               criterion='gini',
                               splitter='best',
                               max_depth=depth[np.argmax(val_score1)],
                               min_samples_leaf=i,
                               min_samples_split=10, 
                               class_weight='balanced'
                              )
    m.fit(model_tr, y_train)
    s = f1_score(y_val, m.predict(model_val), average='macro')
    train_score2.append(f1_score(y_train, m.predict(model_tr), average='macro'))
    val_score2.append(f1_score(y_val, m.predict(model_val), average='macro'))

min_samples_spl = np.linspace(5,50,17).astype('int')
train_score3 = []
val_score3 = []
for i in min_samples_spl:
    m = DecisionTreeClassifier(random_state=0,
                               criterion='gini',
                               splitter='best',
                               max_depth=depth[np.argmax(val_score1)],
                               min_samples_leaf=min_leaf[np.argmax(val_score2)],
                               min_samples_split=i, 
                               class_weight='balanced'
                              )
    m.fit(model_tr, y_train)
    s = f1_score(y_val, m.predict(model_val), average='macro')
    train_score3.append(f1_score(y_train, m.predict(model_tr), average='macro'))
    val_score3.append(f1_score(y_val, m.predict(model_val), average='macro'))

m_opt = DecisionTreeClassifier(random_state=0,
                               criterion='gini',
                               splitter='best',
                               max_depth=depth[np.argmax(val_score1)],
                               min_samples_leaf=min_leaf[np.argmax(val_score2)],
                               min_samples_split=min_samples_spl[np.argmax(val_score3)], 
                               class_weight='balanced'
                              )
m_opt.fit(model_tr, y_train)
mat = confusion_matrix(y_val, m_opt.predict(model_val))
print('optimal score',f1_score(y_val, m_opt.predict(model_val), average='macro'))
print('confusion matrix is: \n',mat)
print('accuracy of each class:\n',[round(mat[0,0]/mat[0,:].sum(),4), round(mat[1,1]/mat[1,:].sum(),4),
                                  round(mat[2,2]/mat[2,:].sum(),4),round(mat[3,3]/mat[3,:].sum(),4),
                                  round(mat[4,4]/mat[4,:].sum(),4)])


# In[4]:


plt.figure()
plt.plot(depth,train_score1,label='train')
plt.plot(depth,val_score1,label='val')
plt.legend()
plt.title('depth vs f1-macro')
print(depth[np.argmax(val_score1)])
print(np.max(val_score1))
pass
plt.figure()
plt.plot(min_leaf,train_score2,label='train')
plt.plot(min_leaf,val_score2,label='val')
plt.legend()
plt.title('min_leaf vs f1-macro')
print(min_leaf[np.argmax(val_score2)])
print(np.max(val_score2))
pass
plt.figure()
plt.plot(min_samples_spl,train_score3,label='train')
plt.plot(min_samples_spl,val_score3,label='val')
plt.legend()
plt.title('min_samples_spl vs f1-macro')
print(min_samples_spl[np.argmax(val_score3)])
print(np.max(val_score3))
pass


# In[5]:


joblib.dump(m_opt,'../input/decision_tree.pkl')


# In[6]:


# Adaboost with tree method
#import warnings
#warnings.filterwarnings("ignore")
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
clf = AdaBoostClassifier(n_estimators=1000, random_state=1)
clf.fit(model_tr, y_train)
real_val_macro = [0]
for real_test_predict in clf.staged_predict(model_val):
    if f1_score(y_val, real_test_predict, average='macro')> np.max(real_val_macro) :
        pred_opt = real_test_predict
    real_val_macro.append(
        f1_score(y_val, real_test_predict, average='macro'))
    
    

n_trees_real = len(clf)
plt.figure()
plt.plot(range(1, n_trees_real + 1),
         real_val_macro[1:], c='black',
         label='SAMME.R')


# In[9]:


real_val_macro = [0]
train_macro = []
for real_test_predict in clf.staged_predict(model_val):
    if f1_score(y_val, real_test_predict, average='macro')> np.max(real_val_macro) :
        pred_opt = f1_score(y_val, real_test_predict, average='macro')
    real_val_macro.append(
        f1_score(y_val, real_test_predict, average='macro'))
for real_train_predict in clf.staged_predict(model_tr):
    train_macro.append(f1_score(y_train, real_train_predict, average='macro'))
    

plt.figure()
plt.plot(range(1, n_trees_real + 1),
         real_val_macro[1:], 
         label='val')
plt.plot(range(1, n_trees_real + 1),
         train_macro,
         label='train')
plt.legend()
plt.title('macro score vs num of decision stump')
adaboost_opt = AdaBoostClassifier(n_estimators=np.argmax(real_val_macro[1:])+1, 
                                  random_state=1)


# In[32]:


np.max(real_val_macro)


# In[29]:


clf_ada_pca = AdaBoostClassifier(n_estimators=715, random_state=1).fit(model_tr,y_train)


# In[64]:


joblib.dump(clf_ada_pca,'../input/model/adaboost1.pkl')


# In[31]:


w2v_adaboost = clf_ada_pca
print('TF-IDF feature: Adaboost, best F1-macro score on validation set:',f1_score(y_val, w2v_adaboost.predict(model_val), average='macro'))
mat = confusion_matrix(y_val, w2v_adaboost.predict(model_val))
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, w2v_adaboost.predict(model_val), average=None))
print('accuracy on validation set:', w2v_adaboost.score(model_val,y_val),'\n\n')


# In[20]:


# Adaboost with tree method with out PCA
#import warnings
#warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
from collections import Counter

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
clf_no_pca = AdaBoostClassifier(n_estimators=1000, random_state=1)
clf_no_pca.fit(model_tr1, y_train)
real_val_macro1 = [0]
train_macro1 = []
for real_test_predict in clf_no_pca.staged_predict(model_val1):
    if f1_score(y_val, real_test_predict, average='macro')> np.max(real_val_macro1) :
        pred_opt1 = real_test_predict
    real_val_macro1.append(
        f1_score(y_val, real_test_predict, average='macro'))
for real_train_predict in clf_no_pca.staged_predict(model_tr1):
    train_macro1.append(f1_score(y_train, real_train_predict, average='macro'))


# In[22]:


plt.figure()
plt.plot(range(1, n_trees_real + 1),
         real_val_macro1[1:], 
         label='val')
plt.plot(range(1, n_trees_real + 1),
         train_macro1,
         label='train')
plt.legend()
plt.title('macro score vs num of decision stump')
adaboost_opt = AdaBoostClassifier(n_estimators=np.argmax(real_val_macro[1:])+1, 
                                  random_state=1)


# In[39]:


np.argmax(real_val_macro1)


# In[40]:


clf_no_pca_opt = AdaBoostClassifier(n_estimators=577, random_state=1)
clf_no_pca_opt.fit(model_tr1, y_train)


# In[42]:


w2v_adaboost = clf_no_pca_opt
print('TF-IDF feature: Adaboost, best F1-macro score on validation set:',f1_score(y_val, w2v_adaboost.predict(model_val1), average='macro'))
mat = confusion_matrix(y_val, w2v_adaboost.predict(model_val1))
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, w2v_adaboost.predict(model_val1), average=None))
print('accuracy on validation set:', w2v_adaboost.score(model_val1,y_val),'\n\n')


# In[25]:


joblib.dump(adaboost_opt,'../input/model/adaboost_nopca.pkl')


# In[45]:


from sklearn.linear_model import LogisticRegression
n_alphas = 41
#alphas = np.exp(np.linspace(-5,10,n_alphas))
C = np.exp(np.linspace(-5,10,n_alphas))
a = np.linspace(0.2,0.7,6)
train_score = [[],[],[],[],[],[]]
val_score = [[],[],[],[],[],[]]
for n,u in enumerate(a):
    for j,i in enumerate(C):
        m = LogisticRegression(penalty='elasticnet',C=i,class_weight='balanced',n_jobs=-1, l1_ratio=u,solver='saga').fit(model_tr,y_train)
        train_pre = m.predict(model_tr)
        val_pre = m.predict(model_val)
        train_score[n].append(f1_score(y_train, train_pre, average='macro'))
        val_score[n].append(f1_score(y_val, val_pre, average='macro'))


# In[60]:


elastic_opt = LogisticRegression(penalty='elasticnet',
                                 C=C[(np.argmax(val_score))%41],
                                 class_weight='balanced',n_jobs=-1, 
                                 l1_ratio=a[(np.argmax(val_score))//41],
                                 solver='saga')
elastic_opt.fit(model_tr,y_train)
print('the best macro score on validation set:',f1_score(y_val, elastic_opt.predict(model_val), average='macro'))
mat = confusion_matrix(y_val, elastic_opt.predict(model_val))
print('confusion matrix is: \n',mat)
print('accuracy of each class:\n',f1_score(y_val, elastic_opt.predict(model_val), average=None))


# In[62]:


joblib.dump(elastic_opt,'../input/model/logistic_elsticnet.pkl')


# In[63]:


train_score = np.array(train_score)
val_score = np.array(val_score)
ax1 = plt.gca()
# randomly choose 50 coefficients in the 9676 variables 
# to plot figure of coefs vs regularizer. 
ax1.plot(C, train_score[(np.argmax(val_score))//41], label='train')
ax1.plot(C, val_score[(np.argmax(val_score))//41],label='val')
ax1.set_xscale('log')
#ax1.set_xlim(ax1.get_xlim()[::-1])  # reverse axis
plt.xlabel('C')
plt.ylabel('macro score')
plt.title('macro score of training set and validation set')
plt.axis('tight')
plt.legend()
plt.show()

