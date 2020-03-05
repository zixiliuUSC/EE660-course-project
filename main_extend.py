import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import logging
from gensim.test.utils import datapath
from gensim import utils
from pathlib import Path
import gensim
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.externals import joblib


# load the data set
df_train = pd.read_csv('./input/train.csv')
train = df_train['review'].to_numpy()
df_y_train = pd.read_csv('./input/y_train.csv')
y_train = df_y_train['rate'].to_numpy()
df_val = pd.read_csv('./input/val.csv')
val = df_val['review'].to_numpy()
df_y_val = pd.read_csv('./input/y_val.csv')
y_val = df_y_val['rate'].to_numpy()

# load embedding model and generate average word2vec vector for every sentence
model = gensim.models.Word2Vec.load('./embedding/word2vec.model')
ave_vec = np.zeros((train.shape[0],50),dtype='float')
word_vectors = model.wv
len(word_vectors.vocab)
# generate word2vec for training set
for k, seq in enumerate(train):
    tokens = gensim.utils.simple_preprocess(seq)
    for i in tokens:
        if i in word_vectors.vocab: 
            ave_vec[k] += model.wv[i]
    ave_vec[k] /= len(tokens)
print(ave_vec.shape)

#generate word2vec for validation set
word_vectors = model.wv
len(word_vectors.vocab)
ave_vec_val = np.zeros((val.shape[0],50),dtype='float')
for k, seq in enumerate(val):
    tokens = gensim.utils.simple_preprocess(seq)
    for i in tokens:
        if i in word_vectors.vocab: 
            ave_vec_val[k] += model.wv[i]
    ave_vec_val[k] /= np.array(len(tokens))
print(ave_vec_val.shape)


w2v_logi_L2 = joblib.load('./input/model_w2v/logistic_L2.pkl')
print('Word2Vec feature: Logistic Regression with L2 penalty, best F1-macro score on validation set:',f1_score(y_val, w2v_logi_L2.predict(ave_vec_val), average='macro'))
mat = confusion_matrix(y_val, w2v_logi_L2.predict(ave_vec_val))
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, w2v_logi_L2.predict(ave_vec_val), average=None))
print('accuracy on validation set:', w2v_logi_L2.score(ave_vec_val,y_val),'\n\n')

w2v_logi_L1 = joblib.load('./input/model_w2v/logistic_L1.pkl')
print('Word2Vec feature: Logistic Regression with L1 penalty, best F1-macro score on validation set:',f1_score(y_val, w2v_logi_L1.predict(ave_vec_val), average='macro'))
mat = confusion_matrix(y_val, w2v_logi_L1.predict(ave_vec_val))
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, w2v_logi_L1.predict(ave_vec_val), average=None))
print('accuracy on validation set:', w2v_logi_L1.score(ave_vec_val,y_val),'\n\n')

w2v_logi_elsticnet = joblib.load('./input/model_w2v/logistic_elastic.pkl')
print('Word2Vec feature: Logistic Regression with Elsticnet, best F1-macro score on validation set:',f1_score(y_val, w2v_logi_elsticnet.predict(ave_vec_val), average='macro'))
mat = confusion_matrix(y_val, w2v_logi_elsticnet.predict(ave_vec_val))
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, w2v_logi_elsticnet.predict(ave_vec_val), average=None))
print('accuracy on validation set:', w2v_logi_elsticnet.score(ave_vec_val,y_val),'\n\n')

w2v_ridge = joblib.load('./input/model_w2v/w2v_ridge.pkl')
val_pre = w2v_ridge.predict(ave_vec_val).astype('int')
val_pre[val_pre > 5] = np.int(5)
val_pre[val_pre < 1] = np.int(1)
print('Word2Vec feature: Ridge Rgression, best F1-macro score on validation set:',f1_score(y_val, val_pre, average='macro'))
mat = confusion_matrix(y_val, val_pre)
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, val_pre, average=None))
print('accuracy on validation set:', accuracy_score(y_val, val_pre),'\n\n')

w2v_adaboost = joblib.load('./input/model_w2v/w2v_adaboost.pkl')
print('Word2Vec feature: Adaboost, best F1-macro score on validation set:',f1_score(y_val, w2v_adaboost.predict(ave_vec_val), average='macro'))
mat = confusion_matrix(y_val, w2v_adaboost.predict(ave_vec_val))
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, w2v_adaboost.predict(ave_vec_val), average=None))
print('accuracy on validation set:', w2v_adaboost.score(ave_vec_val,y_val),'\n\n')

w2v_dec_tr = joblib.load('./input/model_w2v/decision_tree_w2v.pkl')
print('Word2Vec feature: Decision Tree Classifier, best F1-macro score on validation set:',f1_score(y_val, w2v_dec_tr.predict(ave_vec_val), average='macro'))
mat = confusion_matrix(y_val, w2v_dec_tr.predict(ave_vec_val))
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, w2v_dec_tr.predict(ave_vec_val), average=None))
print('accuracy on validation set:', w2v_dec_tr.score(ave_vec_val,y_val),'\n\n')

vectorizer = joblib.load('./input/model/tfidf.pkl')
model_tr = vectorizer.transform(train)
model_val = vectorizer.transform(val)
pca = joblib.load('./input/model/pca.pkl')
model_val = pca.transform(model_val.toarray())
model_tr = pca.transform(model_tr.toarray())

tf_idf_logi_l2 = joblib.load('./input/model/logistic_L2.pkl')
print('TF-IDF feature: Logistic regression with L2 penalty, best F1-macro score on validation set:',
    f1_score(y_val, tf_idf_logi_l2.predict(model_val), average='macro'))
mat = confusion_matrix(y_val, tf_idf_logi_l2.predict(model_val))
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, tf_idf_logi_l2.predict(model_val), average=None))
print('accuracy on validation set:', tf_idf_logi_l2.score(model_val,y_val),'\n\n')

tf_idf_logi_l1 = joblib.load('./input/model/logistic_L1.pkl')
print('TF-IDF feature: Logistic regression with L1 penalty, best F1-macro score on validation set:',
    f1_score(y_val, tf_idf_logi_l1.predict(model_val), average='macro'))
mat = confusion_matrix(y_val, tf_idf_logi_l1.predict(model_val))
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, tf_idf_logi_l1.predict(model_val), average=None))
print('accuracy on validation set:', tf_idf_logi_l1.score(model_val,y_val),'\n\n')

tf_idf_logi_elsticnet = joblib.load('./input/model/logistic_elsticnet.pkl')
print('TF-IDF feature: Logistic regression with Elsticnet, best F1-macro score on validation set:',
    f1_score(y_val, tf_idf_logi_elsticnet.predict(model_val), average='macro'))
mat = confusion_matrix(y_val, tf_idf_logi_elsticnet.predict(model_val))
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, tf_idf_logi_elsticnet.predict(model_val), average=None))
print('accuracy on validation set:', tf_idf_logi_elsticnet.score(model_val,y_val),'\n\n')

tf_idf_ridge = joblib.load('./input/model/tf-idf_ridge.pkl')
val_pre = tf_idf_ridge.predict(model_val).astype('int')
val_pre[val_pre > 5] = np.int(5)
val_pre[val_pre < 1] = np.int(1)
print('TF-IDF feature: Ridge regression, best F1-macro score on validation set:',
    f1_score(y_val, val_pre, average='macro'))
mat = confusion_matrix(y_val, val_pre)
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, val_pre, average=None))
print('accuracy on validation set:', accuracy_score(y_val, val_pre),'\n\n')

tf_idf_adaboost = joblib.load('./input/model/adaboost1.pkl')
val_pre = tf_idf_adaboost.predict(model_val)
print('TF-IDF feature: Adaboost, best F1-macro score on validation set:',
    f1_score(y_val, val_pre, average='macro'))
mat = confusion_matrix(y_val, val_pre)
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, val_pre, average=None))
print('accuracy on validation set:', accuracy_score(y_val, val_pre),'\n\n')

tf_idf_decision_tree = joblib.load('./input/model/decision_tree.pkl')
print('TF-IDF feature: Decision tree Classifier, best F1-macro score on validation set:',
    f1_score(y_val, tf_idf_decision_tree.predict(model_val), average='macro'))
mat = confusion_matrix(y_val, tf_idf_decision_tree.predict(model_val))
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_val, tf_idf_decision_tree.predict(model_val), average=None))
print('accuracy on validation set:', tf_idf_decision_tree.score(model_val,y_val),'\n\n')

print('Test set result: ')
print('optimal model')
df_test = pd.read_csv('./input/test_review.csv')
test = df_test['review'].to_numpy()
df_y_test = pd.read_csv('./input/test_rate.csv')
y_test = df_y_test['rate'].to_numpy()
model_test = vectorizer.transform(test)
model_test = pca.transform(model_test.toarray())
print('TF-IDF feature: Logistic regression with L2 penalty, best F1-macro score on test set:',
    f1_score(y_test, tf_idf_logi_l2.predict(model_test), average='macro'))
mat = confusion_matrix(y_test, tf_idf_logi_l2.predict(model_test))
print('confusion matrix on test set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_test, tf_idf_logi_l2.predict(model_test), average=None))
print('accuracy on test set:', tf_idf_logi_l2.score(model_test,y_test),'\n\n')

print('baseline model')

val_pre = tf_idf_ridge.predict(model_test).astype('int')
val_pre[val_pre > 5] = np.int(5)
val_pre[val_pre < 1] = np.int(1)
print('TF-IDF feature: Ridge regression, best F1-macro score on validation set:',
    f1_score(y_test, val_pre, average='macro'))
mat = confusion_matrix(y_test, val_pre)
print('confusion matrix on validation set is: \n',mat)
print('F1 score of each class:\n',f1_score(y_test, val_pre, average=None))
print('accuracy on validation set:', accuracy_score(y_test, val_pre),'\n\n')