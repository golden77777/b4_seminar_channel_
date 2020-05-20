
# coding: utf-8

# ### 50. データの入手・整形
# News Aggregator Data Setをダウンロードし、以下の要領で学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．
# 
# 1. ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
# 2. 情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
# 3. 抽出された事例をランダムに並び替える．
# 4. 抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ（このファイルは後に問題70で再利用する）．
# 
# 学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．

# In[1]:

import pandas as pd
from sklearn.model_selection import train_test_split

newsCorpora = pd.read_csv('NewsAggregatorDataset/newsCorpora.csv',  #値はタブ区切り、namesで列名を指定
                          names = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'], sep='\t')
#isinメソッドで複数の値にマッチする行を取得
index = newsCorpora['PUBLISHER'].isin(["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"])
df = newsCorpora[index].sample(frac=1) #sampleメソッドでランダムサンプリング（frac=1のとき全ての行）

#8:1:1に分割
train_df, valid_test_df = train_test_split(df, test_size=0.2)
valid_df, test_df = train_test_split(valid_test_df, test_size=0.5)
#txtファイルに保存
train_df.to_csv('train.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
valid_df.to_csv('valid.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
test_df.to_csv('test.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)

#各カテゴリの事例数を確認
print("訓練データ")
train_df['CATEGORY'].value_counts()


# In[2]:

print("検証データ")
valid_df['CATEGORY'].value_counts()


# In[3]:

print("評価データ")
test_df['CATEGORY'].value_counts()


# ### 51. 特徴量抽出
# 学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．

# In[4]:

#単語の出現頻度
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer()
#ベクトル化
X_train = vectorizer.fit_transform(train_df['TITLE'])
X_valid = vectorizer.transform(valid_df['TITLE'])
X_test = vectorizer.transform(test_df['TITLE'])
# スパース行列から密行列に変換
np.savetxt('train.feature.txt', X_train.toarray(), fmt='%d') 
np.savetxt('valid.feature.txt', X_valid.toarray(), fmt='%d')
np.savetxt('test.feature.txt', X_test.toarray(), fmt='%d')


# ### 52. 学習
# 51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．

# In[5]:

from sklearn.linear_model import LogisticRegression

# ロジスティック回帰モデルのインスタンスを作成
lr = LogisticRegression()

Y_train, Y_valid, Y_test = train_df['CATEGORY'], valid_df['CATEGORY'], test_df['CATEGORY']
# ロジスティック回帰モデルの重みを学習
lr.fit(X_train, Y_train)

print("係数：", lr.coef_)
print("切片：", lr.intercept_)


# ### 53. 予測
# 52で学習したロジスティック回帰モデルを用い，与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．

# In[6]:

#カテゴリの予測
Y_test_pred = lr.predict(X_test)
print(Y_test_pred)


# In[7]:

#予測確率
probs = lr.predict_proba(X_test)
print(probs)


# ### 54. 正解率の計測
# 52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ．

# In[8]:

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#学習データからカテゴリ予測
Y_train_pred = lr.predict(X_train)

print("学習データ")
print("正解率：", accuracy_score(y_true=Y_train, y_pred=Y_train_pred))
print("評価データ")
print("正解率：", accuracy_score(y_true=Y_test, y_pred=Y_test_pred))


# ### 55. 混同行列の作成
# 52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，学習データおよび評価データ上で作成せよ．

# In[9]:

print("学習データ")
print(confusion_matrix(y_true=Y_train, y_pred=Y_train_pred))
print("評価データ")
print(confusion_matrix(y_true=Y_test, y_pred=Y_test_pred))


# ### 56. 適合率，再現率，F1スコアの計測
# 52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．カテゴリごとに適合率，再現率，F1スコアを求め，カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ．

# In[10]:

print("【適合率】")
print("micro:", precision_score(y_true=Y_test, y_pred=Y_test_pred, average='micro'))
print("macro:", precision_score(y_true=Y_test, y_pred=Y_test_pred, average='macro'))

print("【再現率】")
print("micro:", recall_score(y_true=Y_test, y_pred=Y_test_pred, average='micro'))
print("macro:", recall_score(y_true=Y_test, y_pred=Y_test_pred, average='macro'))

print("【F1スコア】")
print("micro:", f1_score(y_true=Y_test, y_pred=Y_test_pred, average='micro'))
print("macro:", f1_score(y_true=Y_test, y_pred=Y_test_pred, average='macro'))


# In[11]:

#クラスごと
from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_test_pred))


# ### 57. 特徴量の重みの確認
# 52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．

# In[12]:

names = np.array(vectorizer.get_feature_names())

for c, coef in zip(lr.classes_, lr.coef_):
    #coefの値でソートしたときのインデックスを取得 # [::-1]で降順
    idx = np.argsort(coef)[::-1]
    print('カテゴリ', c)
    print('重みの高い順：', names[idx][:10])
    print('重みの低い順：', names[idx][10:])
    print()


# ### 58. 正則化パラメータの変更
# ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，学習時の過学習（overfitting）の度合いを制御できる．異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．

# In[17]:

import matplotlib.pyplot as plt

train_score = []
valid_score = []
test_score = []
C = np.logspace(-5, 4, 10) #指数関数の生成する値をリストにする

for c in C:
    lr = LogisticRegression(C=c) #Cで正則化の強さを指定
    lr.fit(X_train, Y_train)
    Y_train_pred = lr.predict(X_train)
    Y_valid_pred = lr.predict(X_valid)
    Y_test_pred = lr.predict(X_test)
    train_score.append(accuracy_score(y_true=Y_train, y_pred=Y_train_pred))
    valid_score.append(accuracy_score(y_true=Y_valid, y_pred=Y_valid_pred))
    test_score.append(accuracy_score(y_true=Y_test, y_pred=Y_test_pred))
    
#グラフ表示
plt.rcParams["font.family"] = "Hiragino Sans"

plt.style.use("ggplot")
fig, ax = plt.subplots(1, 1)
ax.plot(C, train_score, label="train_data")
ax.legend(loc="lower right") #凡例を右上に表示

ax.plot(C, valid_score, label="valid_data")
ax.legend(loc="lower right")

ax.plot(C, test_score, label="test_data")
ax.legend(loc="lower right")

plt.ylim([0,1.1])
plt.xscale('log') #x軸をログスケール
plt.xlabel('C')
plt.ylabel('正解率')
plt.title('正則化パラメータと正解率の関係')

plt.show()


# ### 59. ハイパーパラメータの探索
# 学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．

# In[14]:

#ハイパーパラメータ
C = np.logspace(-5, 4, 10)
class_weight = [None, 'balanced'] #クラス数の重みを自動で付ける
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] #学習アルゴリズム

#検証データの正解率の最大値
best_valid_score = 0
best_test_score = 0
best_param = {}

for c in C:
    for w in class_weight:
        for s in solver:
            lr = LogisticRegression(C=c, class_weight=w, solver=s, multi_class='auto')
            lr.fit(X_train, Y_train)
            Y_valid_pred = lr.predict(X_valid)
            Y_test_pred = lr.predict(X_test)
            valid_score = accuracy_score(y_true=Y_valid, y_pred=Y_valid_pred)
            test_score = accuracy_score(y_true=Y_test, y_pred=Y_test_pred)
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_param['C'], best_param['class_weight'], best_param['solver'] = c, w, s
                best_test_score = test_score

print('正解率が最も高くなる学習アルゴリズム・パラメータ：', best_param)
print('評価データ上の正解率：', best_test_score)


# ### 練習

# In[15]:

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ベクトル化する文字列
sample = np.array(['Apple computer of the apple mark', 'linux computer', 'windows computer'])

# TfidfVectorizer
vec_tfidf = TfidfVectorizer()

# ベクトル化
X = vec_tfidf.fit_transform(sample)

print('Vocabulary size: {}'.format(len(vec_tfidf.vocabulary_)))
print('Vocabulary content: {}'.format(vec_tfidf.vocabulary_))

pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())


# In[ ]:




# In[16]:

#tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer #TF（単語の出現頻度）とIDF（単語のレア度）を掛けたもの
import numpy as np

vec_tfidf = TfidfVectorizer()

#行列に変換
train_matrix = np.array(train_df['TITLE'])
valid_matrix = np.array(valid_df['TITLE'])
test_matrix = np.array(test_df['TITLE'])


#ベクトル化
X_train = vec_tfidf.fit_transform(train_matrix)
X_valid = vec_tfidf.fit_transform(valid_matrix)
X_test = vec_tfidf.fit_transform(test_matrix)

np.savetxt('train.feature.txt', X_train.toarray()) # スパース行列から密行列に変換 
np.savetxt('valid.feature.txt', X_valid.toarray(), fmt='%d')
np.savetxt('test.feature.txt', X_test.toarray(), fmt='%d')

