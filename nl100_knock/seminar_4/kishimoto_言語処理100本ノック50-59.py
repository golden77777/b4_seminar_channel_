
# coding: utf-8

# # 第6章:機械学習
# 本章では，Fabio Gasparetti氏が公開しているNews Aggregator Data Setを用い，ニュース記事の見出しを「ビジネス」「科学技術」「エンターテイメント」「健康」のカテゴリに分類するタスク（カテゴリ分類）に取り組む．

# ### 50. データの入手・整形
# News Aggregator Data Setをダウンロードし、以下の要領で学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．
# <pre>
# 1.ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
# 
# 2.情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
# 
# 3.抽出された事例をランダムに並び替える．
# 
# 4.抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ（このファイルは後に問題70で再利用する）．
# </pre>
# 学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．

# In[1]:

import csv
import random
import numpy as np

#ファイル名
news_file_name = "./NewsAggregatorDataset/newsCorpora.csv"
sessions_file_name = "./NewsAggregatorDataset/2pageSessions.csv"

#FORMAT: ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP
#News category: b = business, t = science and technology, e = entertainment, m = health

#今回抽出する情報源のリスト
pub_list = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]

#ファイルの読み込み、カテゴリと見出しの抽出
article_list = []
with open(news_file_name,"r") as f:
    for line in f:
        if line.split("\t")[3] in pub_list:
            article_list.append("\t".join([line.split("\t")[4], line.split("\t")[1]]))

#順番をランダムに並び替える
random.shuffle(article_list)

#nparrayに変換
article_list = np.array(article_list)

#80%を学習データ，残りの10%ずつを検証データと評価データに分割し、txt形式で保存
indexs = [int(article_list.size * n) for n in [0.8, 0.9]]  #0~80%,80~90%,90~100%で分ける
train_data, valid_data, test_data = np.split(article_list, indexs)

np.savetxt("train.txt",train_data, fmt="%s") #デフォルトでfmtは浮動小数点が指定されているので文字列を指定する必要がある
np.savetxt("valid.txt",valid_data, fmt="%s")
np.savetxt("test.txt",test_data, fmt="%s")

#各カテゴリの事例数
def count_category(data):
    b,t,e,m = 0,0,0,0
    for line in data:
        if line.split("\t")[0] == "b":
            b += 1
        if line.split("\t")[0] == "t":
            t += 1
        if line.split("\t")[0] == "e":
            e += 1
        if line.split("\t")[0] == "m":
            m += 1
    return "Business:{}\tScience and Technology:{}\tEntertainment:{}\tHealth:{}".format(b,t,e,m)
    
print("事例数")
print("[学習データ]",count_category(train_data))
print("[検証データ]",count_category(valid_data))
print("[評価データ]",count_category(test_data))


# ### 51. 特徴量抽出
# 学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．

# In[2]:

import nltk
import collections
import csv
#Bag of Words: 単語の出現回数を特徴量とする。bi-gramなども含む

#単語の小文字化、語幹化（ステミング）
def word_standardization(word):
    stemmer = nltk.PorterStemmer()
    word = stemmer.stem(word.lower())
    return word
        
#各データの単語列を変換する
def data_standardization(data):
    data_converted = []
    line_data = []
    words_converted = []
    for line in data:
        line_data.append(line.split("\t")[0])
        for word in line.split("\t")[1].split(" "):
            words_converted.append(word_standardization(word))
        line_data.append(words_converted)
        data_converted.append(line_data)
        line_data = []
        words_converted = []
    return data_converted

train_data_converted = data_standardization(train_data)
valid_data_converted = data_standardization(valid_data)
test_data_converted = data_standardization(test_data)

#学習データに出てくる単語の出現頻度を数える
counter = collections.Counter([ #Counterは辞書型dictのサブクラスで、{要素:出現回数, ...}という形のデータを持つ
    word
    for _, words in train_data_converted
    for word in words
])

# 高頻度・低頻度の語を取り除く
vocab = [
    word
    for word, freq in counter.most_common() #most_common(): (要素, 出現回数)という形のタプルを出現回数順に並べたリストを返す
    if 2 < freq < 300
]

#同様にbi-gramのリストを作る
bi_grams = collections.Counter([
        bi_gram
        for _, words in train_data_converted
        for bi_gram in zip(words, words[1:]) #隣り合う要素をタプルとして受け取る（末尾の足りない分は無視される）
    ]).most_common() #(bi-gramタプル, 出現回数)という形のタプルを出現回数順に並べたリスト

bi_grams = [tupl for tupl, freq in bi_grams if freq > 4] #出現回数が5回以上のbi-gramだけを使う

#見出しごとに単語、bi-gramをカウントしてリストで返す
def count_word(title):
    l = [0]*len(vocab)
    for i,v in enumerate(vocab):
        if v in title:
            l[i] += 1
    return l

def count_bi_gram(title):
    l = [0]*len(bi_grams)
    title_bi_gram = [bi for bi in zip(title,title[1:])]
    for i,v in enumerate(bi_grams):
        if v in title_bi_gram:
            l[i] += 1
    return l
    
#データから特徴量のリストを作成
def make_feature_list(data):
    all_list = []
    for line in data:
        line_list = []
        line_list.append(line[0])
        for num in count_word(line[1]):
            line_list.append(num)
        for num in count_bi_gram(line[1]):
            line_list.append(num)
        all_list.append(line_list)
    return all_list

#結果の出力
with open("train.feature.txt", "w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(make_feature_list(train_data_converted))
with open("valid.feature.txt", "w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(make_feature_list(valid_data_converted))
with open("test.feature.txt", "w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(make_feature_list(test_data_converted))
    
print(bi_grams)


# ### 52. 学習
# 51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．

# In[3]:

#使うデータの用意
def data_prepare(data):
    all_list = []
    for line in data:
        line_list = []
        for num in count_word(line[1]):
            line_list.append(num)
        for num in count_bi_gram(line[1]):
            line_list.append(num)
        all_list.append(line_list)
    return all_list

def category_prepare(data):
    line_list = []
    for line in data:
        line_list.append(line[0])
    return line_list

train_data = data_prepare(train_data_converted)
valid_data = data_prepare(valid_data_converted)
test_data = data_prepare(test_data_converted)
train_category = category_prepare(train_data_converted)
valid_category = category_prepare(valid_data_converted)
test_category = category_prepare(test_data_converted)

#ここから本編
from sklearn.linear_model import LogisticRegression

#学習
model = LogisticRegression(max_iter=1000) #探索回数:1000
model.fit(train_data, train_category)


# ### 53. 予測
# 52で学習したロジスティック回帰モデルを用い，与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．

# In[4]:

import pandas as pd
'''
predict_proba(X):[データ数]行 × [次元数]列の特徴量行列 X を引数にして、各データがそれぞれのクラスに所属する確率を返す

X = [[-1 -1 0][-1- -2 -1][2 5 1]], y = ['0' '0' '1']みたいなデータの場合、
predict_proba(X)は[[0.9 0.1][0.88 0.12][0.01,0.99]]みたいになる

np.argmax(x)
xの最大値のインデックスを返す
二次元配列であれば
axis=0で行ごとに最大値のインデックス
axis=1で列ごとに最大値のインデックス　を返す

x.max()
xの最大値を返す
二次元配列であれば
axis=0で行ごとに最大値
axis=1で列ごとに最大値　を返す
'''
def pred(x):
    x = np.array(x) #argmaxを使うためにnparrayに変換
    p = model.predict_proba(x)
    preds = np.argmax(p,axis=1)
    probs = p.max(axis=1)
    return preds, probs

#学習データを予測
preds_tr, probs_tr = pred(train_data)
pd.DataFrame([[y, p] for y, p in zip(preds_tr, probs_tr)], columns = ['予測', '確率'])

#評価データを予測
preds_ts, probs_ts = pred(test_data)
pd.DataFrame([[y, p] for y, p in zip(preds_ts, probs_ts)], columns = ['予測', '確率'])


# ### 54. 正解率の計測
# 52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ．

# In[5]:

from sklearn.metrics import accuracy_score

'''
predict(x)
[データ数]行 × [次元数]列の特徴量行列 X を引数にして、データ数分の予測ラベルを返す

正解率（accuracy）
すべてのサンプルのうち正解したサンプルの割合
関数accuracy_score()で算出できる
'''
#カテゴリ名を数値化する
def categ_to_num(categ_list):
    return [int(c.replace('b', '0').replace('e', '1').replace('m', '2').replace('t', '3')) for c in categ_list]

print("学習データ")
print(accuracy_score(categ_to_num(train_category), preds_tr))

print("評価データ")
print(accuracy_score(categ_to_num(test_category), preds_ts))


# ### 55. 混同行列の作成
# 52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，学習データおよび評価データ上で作成せよ．

# In[6]:

from sklearn.metrics import confusion_matrix #混同行列を作る
import seaborn as sns #ヒートマップ形式で出力して見やすくする
import matplotlib.pyplot as plt

#学習データ
cm_tr = confusion_matrix(categ_to_num(train_category), preds_tr)

sns.heatmap(cm_tr, annot=True, cmap = 'Blues', xticklabels = ["b","e","m","t"], yticklabels = ["b","e","m","t"])
plt.savefig('confusion_matrix_train.png')

#annot=True　数値を記載
#cmap = 'Blues' 青を基調とする（見やすい）
#xticklabels,yticklabels 軸のラベルを指定する


# In[7]:

#評価データ
cm_ts = confusion_matrix(categ_to_num(test_category), preds_ts)

sns.heatmap(cm_ts, annot=True, cmap = 'Blues', xticklabels = ["b","e","m","t"], yticklabels = ["b","e","m","t"])
plt.savefig('confusion_matrix_test.png')


# ### 56. 適合率，再現率，F1スコアの計測
# 52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．カテゴリごとに適合率，再現率，F1スコアを求め，カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ．

# In[8]:

#precision 適合率 陽性と予測されたサンプルのうち正解したサンプルの割合
#recall 再現率 実際に陽性のサンプルのうち正解したサンプルの割合
#f1 F1スコア 適合率と再現率の調和平均

from statistics import mean

def precision(c):
    result = [0.0]*4
    for i in range(4):
        FP = sum([c[j][i] for j in range(4) if j != i])
        result[i] = (c[i][i]/(c[i][i]+FP))
    return result

def recall(c):
    result = [0.0]*4
    for i in range(4):
        FN = sum([c[i][j] for j in range(4) if j != i])
        result[i] = (c[i][i]/(c[i][i]+FN))
    return result

def f1(c):
    result = [0.0]*4
    for i in range(4):
        result[i] = 2*precision(c)[i]*recall(c)[i]/(precision(c)[i]+recall(c)[i])
    return result

def micro(c):
    result = [0.0]*3 #[precision, recall, f1]
    result[0] = sum([c[i][i] for i in range(4)])/sum([c[i][i] + sum([c[j][i] for j in range(4) if j != i]) for i in range(4)])
    result[1] = sum([c[i][i] for i in range(4)])/sum([c[i][i] + sum([c[i][j] for j in range(4) if j != i]) for i in range(4)])
    result[2] = 2*result[0]*result[1]/(result[0]+result[1])
    return result
    
def macro(c):
    result = [0.0]*3 #[precision, recall, f1]
    result[0] = mean(precision(c))
    result[1] = mean(recall(c))
    result[2] = 2*result[0]*result[1]/(result[0]+result[1])
    return result

print("[ 適合率 ]\tB:{:.4f}\tE:{:.4f}\tM:{:.4f}\tT:{:.4f}\tマイクロ平均:{:.4f}\tマクロ平均:{:.4f}"      .format(precision(cm_ts)[0],precision(cm_ts)[1],precision(cm_ts)[2],precision(cm_ts)[3],micro(cm_ts)[0],macro(cm_ts)[0]))

print("[ 再現率 ]\tB:{:.4f}\tE:{:.4f}\tM:{:.4f}\tT:{:.4f}\tマイクロ平均:{:.4f}\tマクロ平均:{:.4f}"      .format(recall(cm_ts)[0],recall(cm_ts)[1],recall(cm_ts)[2],recall(cm_ts)[3],micro(cm_ts)[1],macro(cm_ts)[1]))

print("[F1スコア]\tB:{:.4f}\tE:{:.4f}\tM:{:.4f}\tT:{:.4f}\tマイクロ平均:{:.4f}\tマクロ平均:{:.4f}"      .format(f1(cm_ts)[0],f1(cm_ts)[1],f1(cm_ts)[2],f1(cm_ts)[3],micro(cm_ts)[2],macro(cm_ts)[2]))


# In[9]:

#簡単な方法
from sklearn.metrics import classification_report

print(classification_report(categ_to_num(test_category), preds_ts))


# ### 57. 特徴量の重みの確認
# 52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．

# In[11]:

vocab.extend(bi_grams)

kategori = ["Business","Entertainment","Health","Science and Technology"]

for categ in range(4):
    dic = {k:v for k, v in zip(vocab, model.coef_[categ])}
    dic_sorted = sorted(dic.items(), key=lambda x:x[1], reverse=True)
    max_feature = dic_sorted[:100]
    min_feature = dic_sorted[-100:]
    min_feature.reverse()
    print("\n",kategori[categ])
    print("\n特徴量\t重み\t(トップ１０0)")
    for i in range(100):
        print("{}\t\t\t{:.4f}".format(max_feature[i][0],max_feature[i][1]))
    print("\n特徴量\t重み\t(ワースト１０0)")
    for i in range(100):
        print("{}\t\t\t{:.4f}".format(min_feature[i][0],min_feature[i][1]))


# ### 58. 正則化パラメータの変更
# ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，学習時の過学習（overfitting）の度合いを制御できる．異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．

# In[131]:

# tqdmでリアルタイムで進捗をみる
from tqdm import tqdm
import time

for i in tqdm(range(20)):
    time.sleep(1)


# In[133]:

bar1 = tqdm(total=10)
for i in range(10):
    bar1.update(1) #毎ターン出てくる
    time.sleep(0.1)
bar1.close()


# In[165]:

import matplotlib.pyplot as plt
import japanize_matplotlib
from tqdm import tqdm



train_accuracy =[]
valid_accuracy =[]
test_accuracy =[]

for num in tqdm(range(1,51)): #進捗をみる
    
    #学習
    model = LogisticRegression(max_iter=1000,C=num*0.1) #探索回数:1000
    model.fit(train_data, train_category)

    #学習データを予測
    preds_tr, probs_tr = pred(train_data)

    #検証データを予測
    preds_vl, probs_vl = pred(valid_data)

    #評価データを予測
    preds_ts, probs_ts = pred(test_data)

    #学習データ正解率
    train_accuracy.append(accuracy_score(categ_to_num(train_category), preds_tr))

    #検証データ正解率
    valid_accuracy.append(accuracy_score(categ_to_num(valid_category), preds_vl))

    #評価データ正解率
    test_accuracy.append(accuracy_score(categ_to_num(test_category), preds_ts))

C_list = [c*0.1 for c in range(1,51)]

plt.plot(C_list, train_accuracy, label = "train")
plt.plot(C_list, valid_accuracy, label = "valid")
plt.plot(C_list, test_accuracy, label = "test")
plt.legend()
plt.xlabel("正則化パラメータ")
plt.ylabel("正解率")
plt.show()


# ### 59. ハイパーパラメータの探索
# 学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．

# In[229]:

#ロジスティック回帰のパラメータを変えてみる

#学習
model = LogisticRegression(max_iter=1000,C=5.0) #結果これが一番正解率高かった

model.fit(train_data, train_category)

#予測
preds, probs = pred(valid_data)

#正解率
print("正解率")
print(accuracy_score(categ_to_num(valid_category), preds))

#正解率が何故か固定されてしまったので不可能



# penalty	l1,l2	ペナルティ関数(正則化の仕方を決める)の種類を設定
# 
# dual	bool型	
# 
# tol	float型	基準を停止するための許容差, 誤差関数の値
# 
# C	float型	正則化の強度
# 
# fit_intercept	bool型	
# 
# intercept_scaling	float型	
# 
# class_weight	辞書型、balanced	クラスに重みをつける. 偏りのあるデータセットに対して予測を行う際に有用
# 
# random_state	int型	データをシャッフルするときの乱数シード値. 設定しておくと毎回同じ結果が得られる.
# 
# solver	newton-cg、lbfgs、liblinear、sag、saga	最適なパラメータの探索方法を設定する. 誤差関数を最小化するパラメータを求める方法
# 
# データセットに対して
# 
# liblinear：小さいデータセットに対していい選択
# 
# sag、saga：大きいデータセットに対し収束が早い
# 
# 多次元問題に対して
# 
# newton-cg、sag、saga、lbfgs：多項式損失関数を扱える
# 
# liblinear：1対他に限られる
# 
# 正則化に対して
# 
# lbfgs、sagはL2正則化に対してのみ使用可能。他は両方可能
# 
# max_iter	int型	収束計算の最大試行回数
# 
# multi_class	ovr、multinomial、auto	**
# 
# ovr：二値分類問題に適している
# 
# multinomial：３値以上の問題に適している
# 
# auto：solver=’liblinear’の場合に’ovr’を選択し、他の場合’multinomial’を選択する
# 
# verbose	int型	
# 
# warm_start	bool型	Trueに指定すると、再学習する際にモデルを初期化せず、既に学習したモデルに更に学習を追加する
# 
# n_jobs	int型、None	multi_class = ‘liblinear’の場合以外で、CPUコアの使用個数を設定. None=1、-1にすると使用可能なすべてのCPUコアを使用する.
# 
# l1_ratio	float、None	

# In[ ]:

#LinearSVC
#エラーが出て正解率0.0と出力された（失敗）


# In[231]:

#NN
from sklearn import neural_network

#学習
model = neural_network.MLPClassifier(activation="tanh", alpha=0.0001)

model.fit(train_data, train_category)

#予測
preds, probs = pred(valid_data)

#正解率
print("正解率")
print(accuracy_score(categ_to_num(valid_category), preds))


# In[232]:

#NN
from sklearn import neural_network

#学習
model = neural_network.MLPClassifier(activation="relu", alpha=0.0001)

model.fit(train_data, train_category)

#予測
preds, probs = pred(valid_data)

#正解率
print("正解率")
print(accuracy_score(categ_to_num(valid_category), preds))


# In[248]:

#K近傍法
from sklearn.neighbors import KNeighborsClassifier

k = int(input("k?"))
#学習
model = KNeighborsClassifier(n_neighbors=k)

model.fit(train_data, train_category)

#予測
preds, probs = pred(valid_data)

#正解率
print("正解率")
print(accuracy_score(categ_to_num(valid_category), preds))


# In[14]:

#SVC
from sklearn import svm

#学習
model = svm.SVC(kernel='poly', gamma=1/2 , C=1.0)

model.fit(train_data, train_category)

#予測
preds, probs = pred(valid_data)

#正解率
print("正解率")
print(accuracy_score(categ_to_num(valid_category), preds))

#動かない

