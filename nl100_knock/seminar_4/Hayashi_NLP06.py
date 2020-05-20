
# coding: utf-8

# ## **第6章: 機械学習**
# 

# In[ ]:

#順番決め
#import random
#l = ["井原","岸本","下川","田嶋","林","日笠"]
#random.shuffle(l)
#print(l)


# In[ ]:

import warnings
warnings.simplefilter("ignore")


# #### **50. データの入手・整形**

# In[ ]:

import random
with open("newsCorpora.csv","r") as f:
    text_list = f.readlines()

with open("new_data.txt","w") as f2:
    new_list = []
    for t in text_list:
        t = t.split("\t")
        if t[3] in ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]:
            new_list.append(t)
    random.shuffle(new_list)
    for n in new_list:
        f2.write("\t".join(n))

#区切る数字を決める
n1 = int(len(new_list)*(4/5))
n2 = int(len(new_list)*(9/10))

with open("train.txt","w") as f3:
    for n in new_list[:n1]:
        f3.write("\t".join([n[1],n[4]])+"\n")        
with open("valid.txt","w") as f4:
    for n in new_list[n1:n2]:
        f4.write("\t".join([n[1],n[4]])+"\n")        
with open("test.txt","w") as f5:
    for n in new_list[n2:]:
        f5.write("\t".join([n[1],n[4]])+"\n")        


# In[ ]:

#行数の確認
get_ipython().system(' wc -l new_data.txt')
get_ipython().system(' wc -l train.txt')
get_ipython().system(' wc -l valid.txt')
get_ipython().system(' wc -l test.txt')


# #### **51. 特徴量抽出**

# 言語処理の流れ→https://qiita.com/Hironsan/items/2466fe0f344115aff177

# In[4]:

#ストップワードをリスト化
import nltk
from nltk.corpus import stopwords
stemmer = nltk.PorterStemmer()
nltk.download('stopwords')
stopword_list = list(set(stopwords.words('english')))
print(len(stopword_list))


# stopword→https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

# In[5]:

import numpy as np
import pandas as pd
import re
import collections

#データをdetaframeに入れえる
df = pd.read_csv('new_data.txt', sep='\t', header=None)
word_list = list()
for i in range(len(df)):
    for word in df[1][i].split():
        word_list.append(re.sub(r":|,|-|'|\(|\)|.*\d.*","",word.lower()))

c = collections.Counter(word_list)
w_list = [i[0] for i in c.items() if i[1] >= 2 and i[0] not in stopword_list]

for i in range(len(df)):
    l = list()
    for word in df[1][i].split():
        if re.sub(r":|,|-|'|\(|\)","",word.lower()) in w_list:
            l.append(stemmer.stem(re.sub(r":|,|-|'|\(|\)|^.*\d.*$","",word.lower())))
    df[1][i] = " ".join(l)



df.head()


# 1度しかでてこない単語の処理→https://qiita.com/kidaufo/items/9865ea50113aa2fc6115

# In[6]:

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(df[1].values)
df_feature = pd.DataFrame(vecs.toarray(),columns=vectorizer.get_feature_names())
df_feature.head()


# TFIDF→https://qiita.com/fujin/items/b1a7152c2ec2b4963160
# TFIDF→http://ailaby.com/tfidf/

# In[7]:

n1 = int(len(df)*(4/5))
n2 = int(len(df)*(9/10))
train_x = df_feature[:n1].values
valid_x = df_feature[n1:n2].values
test_x = df_feature[n2:].values
train_y = df.iloc[:n1,[4]].values.reshape(-1)
valid_y = df.iloc[n1:n2,[4]].values.reshape(-1)
test_y = df.iloc[n2:,[4]].values.reshape(-1)

train_y[:5]


# #### **52. 学習**
# 

# 形態素解析できなかった遺産↓

# In[3]:

#! mkdir tree-tagger
#! cd cmd/
#! wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.tar.gz
#! wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz
#! wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english-par-linux-3.2-utf8.bin.gz
#! wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/install-tagger.sh

#! chmod u+x install-tagger.sh
#! ./install-tagger.sh


# In[21]:

#! pip install treetaggerwrapper


# In[ ]:

#import treetaggerwrapper as ttw
#tagger = ttw.TreeTagger(TAGLANG='en')
#tags = tagger.TagText('I have a pen.')
#print(tags)


# treeTagger関連→https://qiita.com/3000manJPY/items/1c553a89b2c70edaa960,http://otani0083.hatenablog.com/entry/2013/10/01/195037,http://miner.hatenablog.com/entry/572

# In[24]:

# ロジスティック回帰の実行
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0)
lr.fit(train_x, train_y)


# 
# #### **53. 予測**

# In[25]:

lr.predict_proba(valid_x)


# #### **54. 正解率の計測**

# In[ ]:

print('train acc: %.3f' % lr.score(train_x, train_y))
print('test acc: %.3f' % lr.score(test_x, test_y))


# #### **55. 混同行列の作成**

# In[26]:

from sklearn.metrics import confusion_matrix,precision_score
train_y_pred = lr.predict(train_x)
train_cm = confusion_matrix(train_y,train_y_pred)
train_cm


# In[28]:

valid_y_pred = lr.predict(valid_x)
valid_cm = confusion_matrix(valid_y,valid_y_pred)
valid_cm


# #### **56. 適合率，再現率，F1スコアの計測**

# In[29]:

def accuracy(cm,category_num):
    return (sum([sum([num for i,num in enumerate(v) if i!=category_num]) for j,v in enumerate(cm) if j!=category_num])+cm[category_num,category_num])/sum(sum(cm))

def precion(cm,category_num):
    return cm[category_num][category_num]/sum(cm[category_num])    

def recall(cm,category_num):
    return cm[:,category_num][category_num]/sum(cm[:,category_num])

def f1(cm,category_num):
    return (2*precion(cm,category_num)*recall(cm,category_num))/(precion(cm,category_num)+recall(cm,category_num))

for i in range(4):
    print("*",i)
    print('accuracy: %.3f' % accuracy(valid_cm,i))
    print('precion: %.3f' % precion(valid_cm,i))
    print('recall: %.3f' % recall(valid_cm,i))
    print('f1: %.3f' % f1(valid_cm,i))

print("マクロ平均")
#print('precion: %.3f' % (sum([precion(valid_cm,i) for i in range(4)])/4))
#print('recall: %.3f' % (sum([recall(valid_cm,i) for i in range(4)])/4))
#print('f1: %.3f' % (sum([f1(valid_cm,i) for i in range(4)])/4))
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print('accuracy: %.3f' % accuracy_score(valid_y,valid_y_pred))
print('precison: %.3f' % precision_score(valid_y, valid_y_pred,average='macro'))
print('recall: %.3f' % recall_score(valid_y, valid_y_pred,average='macro'))
print('f1: %.3f' % f1_score(valid_y, valid_y_pred,average='macro'))

print("マイクロ平均")
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print('accuracy: %.3f' % accuracy_score(valid_y,valid_y_pred))
print('precison: %.3f' % precision_score(valid_y, valid_y_pred,average='micro'))
print('recall: %.3f' % recall_score(valid_y, valid_y_pred,average='micro'))
print('f1: %.3f' % f1_score(valid_y, valid_y_pred,average='micro'))


# #### **57. 特徴量の重みの確認**

# In[30]:

def print_weight(category_num):
    print("<",category_num,">")
    corf_dict = dict()
    for i,c in enumerate(lr.coef_[category_num]):
        corf_dict[list(df_feature)[i]] = c
    print("*重要度の低い10個")
    for a,b in sorted(corf_dict.items(),key=lambda x:abs(x[1]))[:10]:
        print("{}:{:.3f}".format(a,b))
    print("重要度0の値:",len([i for i,c in corf_dict.items() if c==0]))
    print("*重要度の重い10個")
    for c,d in sorted(corf_dict.items(),key=lambda x:abs(x[1]))[-10:]:
        print("{}:{:.3f}".format(c,d))

for i in range(4):
    print_weight(i)


# 3→health
# 4→science
# 1→business
# 2→entertainment

# #### **58. 正則化パラメータの変更**

# In[31]:

# ロジスティック回帰の実行
from sklearn.linear_model import LogisticRegression
train_score_list = []
test_score_list = []
valid_score_list = []
c_list = [0.01,0.1,1.0,10.0,100.0]
for c in c_list:
    lr = LogisticRegression(C=c)
    lr.fit(train_x, train_y)
    #正解率の計測
    print('*C=',c)
    print('train acc: %.3f' % lr.score(train_x, train_y))
    print('test acc: %.3f' % lr.score(test_x, test_y))
    print('valid acc: %.3f' % lr.score(valid_x, valid_y))
    train_score_list.append(lr.score(train_x, train_y))
    test_score_list.append(lr.score(test_x, test_y))
    valid_score_list.append(lr.score(valid_x, valid_y))


# In[33]:

import matplotlib.pyplot as plt
plt.plot(c_list, train_score_list, label="train",marker=".")
plt.plot(c_list, valid_score_list, label="valid",marker=".")
plt.plot(c_list, test_score_list, label="test",marker=".")

plt.xscale("log")
plt.legend()
plt.show()


# #### **59. ハイパーパラメータの探索**
# 

# SVM

# In[ ]:

#線形
from sklearn.svm import SVC
linear_svm = SVC(kernel='linear', C=0.1)
linear_svm.fit(train_x, train_y)
linear_svm.score(train_x,train_y)


# In[ ]:

linear_svm.score(valid_x,valid_y)


# In[ ]:

#カーネル法
from sklearn.svm import SVC
rbf_svm = SVC(kernel='rbf', gamma=0.1, C=10)
rbf_svm.fit(train_x, train_y)
rbf_svm.score(train_x,train_y)


# In[ ]:

rbf_svm.score(valid_x,valid_y)


# In[ ]:

#カーネル法
from sklearn.svm import SVC
rbf_svm = SVC(kernel='rbf', gamma=0.1, C=100)
rbf_svm.fit(train_x, train_y)
rbf_svm.score(train_x,train_y)


# In[ ]:

rbf_svm.score(valid_x,valid_y)


# In[ ]:

#カーネル法
from sklearn.svm import SVC
rbf_svm = SVC(kernel='rbf', gamma=0.1, C=1)
rbf_svm.fit(train_x, train_y)
rbf_svm.score(train_x,train_y)


# In[ ]:

rbf_svm.score(valid_x,valid_y)


# ランダムフォレスト

# In[ ]:

from sklearn.ensemble import RandomForestClassifier
#ランダムフォレストのインスタンスを生成
rfc_1 = RandomForestClassifier(random_state=0, n_estimators=10)
#モデルを学習
rfc_1.fit(train_x,train_y)

rfc_1.score(train_x,train_y)


# In[ ]:

rfc_1.score(valid_x,valid_y)


# SVMカーネル法

# In[ ]:

param_grid = {'C': [0.1, 1.0, 10, 100],
              'gamma': [0.01, 0.1, 1]}

from sklearn.model_selection import StratifiedKFold
kf_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# GridSearchCVのインスタンスを生成
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# (モデルのインスタンス, 試したいパラメータの値, 分割方法)
gs_svc = GridSearchCV(SVC(), param_grid, cv=kf_5)
gs_svc.fit(train_x,train_y)
# test精度の平均が最も高かった組み合わせを出力
gs_svc.best_params_


# In[ ]:

print('train acc: %.3f' % gs_svc.score(train_x, train_y))
print('test acc: %.3f' % gs_svc.score(test_x, test_y))
print('valid acc: %.3f' % gs_svc.score(valid_x, valid_y))


# *   各カテゴリーの個数を確認していない
# *   validデータも一緒に前処理をしている→致命的
# *   bigramとか形態素解析したかった
# *   重い
# 
# 
