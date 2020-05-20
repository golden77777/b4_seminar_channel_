
# coding: utf-8

# ## 第6章: 機械学習
# 本章では，Fabio Gasparetti氏が公開しているNews Aggregator Data Setを用い，ニュース記事の見出しを「ビジネス」「科学技術」「エンターテイメント」「健康」のカテゴリに分類するタスク（カテゴリ分類）に取り組む

# ### 50. データの入手・整形
# News Aggregator Data Setをダウンロードし、以下の要領で学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．
# 
# 1.ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
# 
# 2.情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
# 
# 3.抽出された事例をランダムに並び替える．
# 
# 4.抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ（このファイルは後に問題70で再利用する）．
# 学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．

# In[ ]:

#50
import random
import zipfile
from sklearn.model_selection import train_test_split

zipfile_name = "NewsAggregatorDataset.zip"
csvfile_name = "newsCorpora.csv"
publisher =['Reuters','Huffington Post','Businessweek','Contactmusic.com','Daily Mail']

with zipfile.ZipFile(zipfile_name,'r') as f:
    with f.open(csvfile_name,'r') as  g:
        data = g.read()

a = data.decode('utf-8')
b = a.split('\n')
c = [b[i].split('\t') for i in range(len(b))]
print(c[0])
d = []
for i in range(len(c)):
    if len(c[i]) > 3 :
        if c[i][3] in publisher:
            d.append([c[i][1],c[i][4]])

train_data,surplus_data =train_test_split(d, test_size=0.2)
valid_data,test_data = train_test_split(surplus_data,test_size = 0.5)

print('学習データ数',len(train_data))
print('検証データ数',len(valid_data))
print('評価データ数',len(test_data))


def write_txt(output_file,data):
    with open(output_file,'w',encoding = 'utf-8') as f:
        for i in range(len(data)-1):
            f.write('\t'.join(data[i])+'\n')
        f.write('\t'.join(data[-1]))
    
write_txt('test.txt',test_data)
write_txt('valid.txt',valid_data)
write_txt('train.txt',train_data)


# ### 51. 特徴量抽出
# 学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．

# In[160]:

#51
with open('stopwords.txt') as f:
    a = f.readlines()
    words=[]
    for i in range(len(a)):
        words.append(a[i].rstrip('\n'))
    
from collections import Counter
words_list = []
for i in range(len(d)):
    word = d[i][0].split()
    for j in range(len(word)):
        w = word[j].lower()
        if w not in words and w.isalpha():
            words_list.append(w)
count = Counter(words_list)
a = count.most_common()
feature_words = []
for i in range(len(a)):
    feature_words.append(a[i][0])

import numpy as  np

def count_words(feature,title):
    vocab_dict = {x:n for n,x in enumerate(feature)}
    lst = [0]*len(feature)
    for words in title:
        if words in vocab_dict:
            lst[vocab_dict[words]] += 1
    return lst

def make_dataset(data):
    lab = [categories.index(data[i][-1]) for i in range(len(data))]
    x = []
    for i in range(len(data)):
        x.append(count_words(feature_words,data[i][0].split()))
    return np.array(x),np.array(lab)

def write_dataset(file_name,xs,ts):
    with open(file_name,'w') as f:
        for t,x in zip(ts,xs):
            line = categories[t]+'\t'+'\t'.join([str(int(n)) for n in x])
            f.write(line+'\n')

train_x, train_t = make_dataset(train_data)
valid_x, valid_t = make_dataset(valid_data)
test_x, test_t = make_dataset(test_data)
write_dataset('train.feature.txt',train_x,train_t)
write_dataset('valid.feature.txt',valid_x,valid_t)
write_dataset('test.feature.txt',test_x,test_t)


# ### 52. 学習
# 51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．

# In[154]:

#52
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 1,max_iter = 1000)
lr.fit(train_x,train_t)


# In[148]:

import numpy as np
from matplotlib import pyplot as plt


# ### 53. 予測
# 52で学習したロジスティック回帰モデルを用い，与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．

# In[158]:

#53
def predict(x):
    prediction = lr.predict_proba(x).argmax(axis=1)
    #インデックス番号を返す
    probability = lr.predict_proba(x).max(axis=1)
    #値を返す
    return prediction, probability
preds, probs = predict(test_x)
pd.DataFrame([(categories[x], y) for x, y in zip(preds, probs)], columns = ['予測カテゴリ', '予測確率'])


# ### 54. 正解率の計測
# 52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ．

# In[159]:

#54
from sklearn.metrics import accuracy_score
def accuracy(lr,t,x):
    ac_score = accuracy_score(t, lr.predict(x))
    return ac_score
print('trainデータ')
print(accuracy(lr,train_t,train_x))
print('testデータ')
print(accuracy(lr,test_t,test_x))


# ### 55. 混同行列の作成
# 52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，学習データおよび評価データ上で作成せよ．

# In[104]:

# 55
from sklearn.metrics import confusion_matrix
def make_confusion_matrix(lr,t,x):
    return  confusion_matrix(t, lr.predict(x))
print('学習データ\n',make_confusion_matrix(lr,train_t,train_x))
print('訓練データ\n',make_confusion_matrix(lr,test_t,test_x))


# ### 56. 適合率，再現率，F1スコアの計測
# 52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．カテゴリごとに適合率，再現率，F1スコアを求め，カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ．

# In[109]:

#56
from sklearn.metrics import precision_score, recall_score, f1_score
pr1 = precision_score(test_t, lr.predict(test_x),average = None)
pr2 = precision_score(test_t, lr.predict(test_x),average = 'micro')
pr3 = precision_score(test_t, lr.predict(test_x),average = 'macro')
pr = np.hstack((pr1,pr2,pr3))
re1 = recall_score(test_t, lr.predict(test_x),average = None)
re2 = recall_score(test_t, lr.predict(test_x),average = 'micro')
re3 = recall_score(test_t, lr.predict(test_x),average = 'macro')
re = np.hstack((re1,re2,re3))
f1 = f1_score(test_t, lr.predict(test_x),average = None)
f2 = f1_score(test_t, lr.predict(test_x),average = 'micro')
f3 = f1_score(test_t, lr.predict(test_x),average = 'macro')
f = np.hstack((f1,f2,f3))

df = pd.DataFrame({'適合率':pr,
                   '再現率':re,
                   'F1スコア':f},
                  index = categories + ['マイクロ平均'] + ['マクロ平均'])
df


# ### 57. 特徴量の重みの確認
# 52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．

# In[155]:

#57
for i in range(len(category_names)):
    index = lr.coef_[i].argsort()[::-1][:10]
    ranking = np.array(feature_words)[index]
    weight = lr.coef_[i][index]
    print(category_names[i])
    display(pd.DataFrame([ranking,weight],index=['特徴量','重み'],columns=np.arange(10)+1))


# In[156]:

#57
for i in range(len(category_names)):
    index = lr.coef_[i].argsort()[::1][:10]
    ranking = np.array(feature_words)[index]
    weight = lr.coef_[i][index]
    print(category_names[i])
    display(pd.DataFrame([ranking,weight],index=['特徴量','重み'],columns=np.arange(10)+1))


# ### 58. 正則化パラメータの変更
# ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，学習時の過学習（overfitting）の度合いを制御できる．異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．

# In[150]:

#58
import matplotlib.pyplot as plt
Cs = np.arange(1,100,2)
lrs = [LogisticRegression(C=C, max_iter=1000).fit(train_x, train_t) for C in Cs]
train_accs = [accuracy(lr, train_t, train_x) for lr in lrs]
valid_accs = [accuracy(lr, valid_t, valid_x) for lr in lrs]
test_accs = [accuracy(lr, test_t, test_x) for lr in lrs]
plt.plot(Cs, train_accs, label = '学習')
plt.plot(Cs, valid_accs, label = '検証')
plt.plot(Cs, test_accs, label = '評価')
plt.legend()
plt.show()


# ### 59. ハイパーパラメータの探索
# 学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．

# In[129]:

#59
max_score = (0,0)
max_score2 = (0,0)
#ロジスティック回帰分析
C_candidate = np.arange(0.1,5.1,0.1)
for c in C_candidate:
    clf = LogisticRegression(penalty='l2', solver='lbfgs', random_state=0, C=c)
    clf.fit(train_x, train_t)
    if max_score[0] < float(accuracy_score(test_t,clf.predict(test_x))):
        max_score = (float(accuracy_score(test_t,clf.predict(test_x))),c)

#SVM
from sklearn import svm
C_candidate = np.arange(0.1,5.1,0.1)
for c in C_candidate:
    clf = svm.LinearSVC(loss='hinge',C=c,class_weight = 'balanced',random_state=0)
    clf.fit(train_x,train_t)
    if max_score2[0] < float(accuracy_score(test_t,clf.predict(test_x))):
        max_score2 = (float(accuracy_score(test_t,clf.predict(test_x))),m)


if max_score2[0] < max_score[0]:
    bestAlg = 'LogisticRegression'
    bestParam = max_score[1]
else:
    bestAlg = 'SVM'
    bestParam = max_score2[1]

print(bestAlg, bestParam)

