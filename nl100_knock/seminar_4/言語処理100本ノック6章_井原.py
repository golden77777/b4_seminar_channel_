
# coding: utf-8

# # 第6章: 機械学習

# 本章では，Fabio Gasparetti氏が公開しているNews Aggregator Data Setを用い，ニュース記事の見出しを「ビジネス」「科学技術」「エンターテイメント」「健康」のカテゴリに分類するタスク（カテゴリ分類）に取り組む．

# 50. データの入手・整形
# 
# News Aggregator Data Setをダウンロードし、以下の要領で学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．
# 
# 1.ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
# 
# 2.情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
# 
# 3.抽出された事例をランダムに並び替える．
# 
# 4.抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ（このファイルは後に問題70で再利用する）．
# 
# 学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．

# In[69]:

#import requests

#url="https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip"
#res=requests.get(url)

import random
import sys
import numpy as np
import pandas as pd
import collections

def tab_split(text):
    return(text.split('\t'))

def space_split(text):
    return(text.split(' '))

def file_write(data_list,file_name):
    with open(file_name,'w',encoding='UTF-8') as f:
        for i in range(len(data_list)):
            data_list[i]='\t'.join(data_list[i])
        f.write('\n'.join(data_list))
            
def file_read(file_name):
    with open (file_name,'r',encoding='UTF-8') as f:
        res = list(map(tab_split,f.read().split('\n')))
    return(res)

def category_counter(data):
    data=np.array(data)
    print(collections.Counter(data[:,0]))

with open('./NewsAggregatorDataset/newsCorpora.csv','r',encoding="UTF-8") as f:
    text = list(map(tab_split,f.read().split('\n')))
    
publisher_list=['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
category_list=['b','t','e','m']
          
print(text[:10])

dataset = list()
for line in text:
    try:
        if (line[3] in publisher_list) and (line[4] in category_list):
            dataset.append([line[4],line[1]])
    except IndexError:
        pass

random.shuffle(dataset)
length = len(dataset)
file_write(dataset[:int(0.8*length)],'./train.txt')
file_write(dataset[int(0.8*length):int(0.9*length)],'./valid.txt')
file_write(dataset[int(0.9*length):],'./test.txt')

category_counter(dataset[:int(0.8*length)])
category_counter(dataset[int(0.8*length):int(0.9*length)])
category_counter(dataset[int(0.9*length):])


# 51. 特徴量抽出
# 
# 学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．

# In[114]:

from sklearn.feature_extraction.text import TfidfVectorizer

def create_feature(file_name,new_file_name):
    text=file_read(file_name)
    data=list()
    for line in text:
        data.append(line[1])
    sample = np.array(data)
    data=list(map(space_split,data))
    # TfidfVectorizer
    vec_tfidf = TfidfVectorizer()
    # ベクトル化
    X = vec_tfidf.fit_transform(sample)
    #print('Vocabulary size: {}'.format(len(vec_tfidf.vocabulary_)))
    #print('Vocabulary content: {}'.format(vec_tfidf.vocabulary_))
    #print(pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names()))
    #file_write(X.toarray(),new_file_name)
    return(X.toarray(),vec_tfidf.get_feature_names())
    

xdata,xdict=create_feature('./train.txt','./train.feature.txt')
ydata,ydict=create_feature('./valid.txt','./valid.feature.txt')
zdata,zdict=create_feature('./test.txt','./test.feature.txt')


# 52. 学習
# 
# 51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．

# In[120]:

import sklearn
from sklearn.linear_model import LogisticRegression

#==============================================
train_data=xdata
print(train_data)

text=file_read('./train.txt')
train_label=np.array(text)[:,0]
print(train_label)
#===============================================
valid_data=np.zeros((ydata.shape[0],len(xdict)))
for i in range(len(ydict)):
    if ydict[i] in xdict:
        valid_data[:,xdict.index(ydict[i])]=ydata[:,i]


text=file_read('./valid.txt')
valid_label=np.array(text)[:,0]
print(valid_label)
#===============================================
test_data=np.zeros((zdata.shape[0],len(xdict)))
for i in range(len(zdict)):
    if zdict[i] in xdict:
        test_data[:,xdict.index(zdict[i])]=zdata[:,i]


text=file_read('./test.txt')
test_label=np.array(text)[:,0]
print(test_label)
#===============================================


model =LogisticRegression(C=1.0,penalty='l2',solver='lbfgs',max_iter=100)
model.fit(train_data,train_label)


# 53. 予測
# 
# 52で学習したロジスティック回帰モデルを用い，与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．

# In[125]:

train_predict = model.predict(train_data)
test_predict = model.predict(test_data)
#train_data*model.coef_


# 54. 正解率の計測
# 
# 52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ．

# In[127]:

from sklearn.metrics import accuracy_score
print(' accuracy : ',accuracy_score(train_label, train_predict))
print(' accuracy : ',accuracy_score(test_label, test_predict))


# 55. 混同行列の作成
# 
# 52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，学習データおよび評価データ上で作成せよ．

# In[128]:

from sklearn.metrics import confusion_matrix
print(confusion_matrix(train_label, train_predict))
print(confusion_matrix(test_label, test_predict))


# 56. 適合率，再現率，F1スコアの計測
# 
# 52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．カテゴリごとに適合率，再現率，F1スコアを求め，カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ．

# In[130]:

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print( ' precision : ' , precision_score(train_label, train_predict,average='micro') )
print( ' recall : ' , recall_score(train_label, train_predict,average='micro') )
print(' f1-score : ',f1_score(train_label, train_predict,average='micro'))

print( ' precision : ' , precision_score(train_label, train_predict,average='macro') )
print( ' recall : ' , recall_score(train_label, train_predict,average='macro') )
print(' f1-score : ',f1_score(train_label, train_predict,average='macro'))


print( ' precision : ' , precision_score(test_label, test_predict,average='micro') )
print( ' recall : ' , recall_score(test_label, test_predict,average='micro') )
print(' f1-score : ',f1_score(test_label, test_predict,average='micro'))


print( ' precision : ' , precision_score(test_label, test_predict,average='macro') )
print( ' recall : ' , recall_score(test_label, test_predict,average='macro') )
print(' f1-score : ',f1_score(test_label, test_predict,average='macro'))


# 57. 特徴量の重みの確認
# 
# 52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．

# In[146]:

print(model.coef_)

category_list=["business","entertainment","health","science and technology"]

for i,vec in enumerate(model.coef_):
    vec=vec.tolist()
    new_vec=sorted(vec,reverse=True)
    print("============="+category_list[i]+" top10==============")
    for k in range(10):
        print("    {} : {:.3f}".format(xdict[vec.index(new_vec[k])],new_vec[k]))
    
    print("============="+category_list[i]+" worst10==============")
    for k in range(10):
        print("    {} : {:.3f}".format(xdict[vec.index(new_vec[-k-1])],new_vec[-k-1]))
    


# 58. 正則化パラメータの変更
# 
# ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，学習時の過学習（overfitting）の度合いを制御できる．異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．

# In[157]:

c=np.arange(0.1,5.01,0.1)
train_result=list()
valid_result=list()
test_result=list()
for i in range(50):
    
    model =LogisticRegression(C=(i+1)/10,penalty='l2',solver='newton-cg',max_iter=100)

    # 学習
    model.fit(train_data, train_label)
    #予測
    train_predict = model.predict(train_data)
    valid_predict = model.predict(valid_data)
    test_predict = model.predict(test_data)

    train_result.append(accuracy_score(train_label, train_predict))
    valid_result.append(accuracy_score(valid_label, valid_predict))
    test_result.append(accuracy_score(test_label, test_predict))
    
import matplotlib.pyplot as plt
train_result=np.array(train_result)
valid_result=np.array(valid_result)
test_result=np.array(test_result)
plt.plot(c,train_result)
plt.plot(c,valid_result)
plt.plot(c,test_result)
plt.show()


# 59. ハイパーパラメータの探索
# 
# 学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．

# In[158]:

from sklearn.model_selection import GridSearchCV
# グリッドサーチ
parameters = {'C': [ 0.2, 0.4, 0.6, 0.8, 1.0 ],
    'penalty' : [ 'l2' ],
    'solver' : [ 'newton-cg' ],
    'max_iter' : [ 100,200 ]
}

# ロジスティック回帰
model = GridSearchCV(LogisticRegression(), parameters, cv=5)

# 学習
model.fit(train_data, train_label)

# 最良モデル
best_model = model.best_estimator_
print( "¥n [ 最良なパラメータ ]" )
print( model.best_params_ )

# 予測
predict = best_model.predict(test_data)

# 予測結果の表示
print( ' accuracy : ' , accuracy_score(test_label, predict) )
print( confusion_matrix(test_label, predict) )


# In[ ]:



