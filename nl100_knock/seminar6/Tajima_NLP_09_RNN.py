
# coding: utf-8

# ### 80. ID番号への変換
# 問題51で構築した学習データ中の単語にユニークなID番号を付与したい．学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，学習データ中で2回以上出現する単語にID番号を付与せよ．そして，与えられた単語列に対して，ID番号の列を返す関数を実装せよ．ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ．

# In[313]:

#50 データの入手・整形
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = []

with open('NewsAggregatorDataset/newsCorpora.csv', encoding='utf-8') as f:
    for line in f:
        columns = line.strip('\n').split('\t')
        dataset.append(columns)

newsCorpora = pd.DataFrame(dataset, columns=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

#isinメソッドで複数の値にマッチする行を取得
index = newsCorpora['PUBLISHER'].isin(["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"])
df = newsCorpora[index].sample(frac=1) #sampleメソッドでランダムサンプリング（frac=1のとき全ての行）

#8:1:1に分割
train_df, valid_test_df = train_test_split(df, test_size=0.2)
valid_df, test_df = train_test_split(valid_test_df, test_size=0.5)
#カテゴリとタイトル名を抽出する
train_df = train_df[['CATEGORY', 'TITLE']].reset_index(drop=True)
test_df = test_df[['CATEGORY', 'TITLE']].reset_index(drop=True)


# In[315]:

#中身の確認
train_df


# In[415]:

test_df


# In[317]:

import re
import nltk
from nltk.corpus import stopwords

#テキストクリーニング
#https://qiita.com/kakiuchis/items/9149c7249f748668bb9b

#ストップワードのリスト化
stemmer = nltk.PorterStemmer()
stopword_list = list(set(stopwords.words('english')))
print('ストップワードの一覧')
print(stopword_list)

#分かち書き
def word_tokenaize(text):
    morphs = nltk.word_tokenize(text)
    return morphs

#小文字化、ステミング（語幹化）
def word_normalization(word):
    word = word.lower()
    return stemmer.stem(word)


# In[319]:

#ワードリストへの追加と系列の長さの最大値を取得（#83のパディング用）
word_list = []
max_len = 0

for i in range(len(train_df)):
    s = re.sub(r"[!\"#\$%&'\(\)-=~\^\|@`\{\}\[\]:\*;\+,<\.>?/\d]", '', train_df['TITLE'][i]) #記号の削除
    words = word_tokenaize(s)
    for word in words:
        norm = word_normalization(word)
        if not norm in stopword_list: #ストップワードを削除
            word_list.append(norm)
        if len(words) > max_len:
            max_len = len(words)
            
    
print(word_list)


# In[363]:

#系列の長さの最大値
print(max_len)


# In[323]:

#80
#単語数のカウント（2回未満は0）
from collections import Counter
c = Counter(word_list)

word_dict = {}

#単語にID番号を付加
idx = 1
for k, v in c.most_common():
    if v >= 2:
        word_dict[k] = idx
        idx += 1
    else:
        word_dict[k] = 0

print('語彙数', len(word_dict))
print(word_dict)


# In[383]:

import torch

#ID番号の列を返す関数
def sent_to_ids(text):
    id_list = []
    s = re.sub(r"[!\"#\$%&'\(\)-=~\^\|@`\{\}\[\]:\*;\+,<\.>?/\d]", '', text) #記号の削除
    words = word_tokenaize(s)
    for word in words:
        norm = word_normalization(word)
        if not norm in stopword_list and norm in word_dict.keys(): #ストップワードを削除 #単語がword_dictに存在するとき
            id_list.append(word_dict[norm])
    #パディング
    for j in range(max_len - len(id_list)):
        id_list.insert(0, 0)
    return torch.as_tensor(id_list, dtype=torch.int64)

print(sent_to_ids('Fed official says weak data caused by weather, should not slow taper'))


# ### 81. RNNによる予測
# ID番号で表現された単語列x=(x1,x2,…,xT)がある．ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）．再帰型ニューラルネットワーク（RNN: Recurrent Neural Network）を用い，単語列xからカテゴリyを予測するモデルとして，次式を実装せよ．
# 
# h→0=0
# 
# h→t=RNN−→−−(emb(xt),h→t−1)
# 
# y=softmax(W(yh)h→T+b(y))
# 
# 
# ただし，emb(x)∈ℝdwは単語埋め込み（単語のone-hot表記から単語ベクトルに変換する関数），h→t∈ℝdhは時刻tの隠れ状態ベクトル，RNN−→−−(x,h)は入力xと前時刻の隠れ状態hから次状態を計算するRNNユニット，W(yh)∈ℝL×dhは隠れ状態ベクトルからカテゴリを予測するための行列，b(y)∈ℝLはバイアス項である（dw,dh,Lはそれぞれ，単語埋め込みの次元数，隠れ状態ベクトルの次元数，ラベル数である）．RNNユニットRNN−→−−(x,h)には様々な構成が考えられるが，典型例として次式が挙げられる．
# 
# RNN−→−−(x,h)=g(W(hx)x+W(hh)h+b(h))
# 
# 
# ただし，W(hx)∈ℝdh×dw，W(hh)∈ℝdh×dh,b(h)∈ℝdhはRNNユニットのパラメータ，gは活性化関数（例えばtanhやReLUなど）である．
# 
# なお，この問題ではパラメータの学習を行わず，ランダムに初期化されたパラメータでyを計算するだけでよい．次元数などのハイパーパラメータは，dw=300,dh=50など，適当な値に設定せよ（以降の問題でも同様である）．

# In[384]:

#単語ベクトルに変換（embedding）
import torch.nn as nn

v_size = len(word_dict)
e_size = 300
h_size = 50

embed = nn.Embedding(v_size, e_size) #300次元の単語ベクトルに変換
s1 =  'Fed official says weak data caused by weather, should not slow taper'
#embeddingの例
sentence_matrix = torch.as_tensor(sent_to_ids(s1), dtype=torch.int64)
# 出力はfloat32のTensor
output = embed(sentence_matrix)
print(output)


# In[385]:

import torch.nn as nn

v_size = len(word_dict)
e_size = 300
h_size = 50
c_size = 4

#nn.Moduleを継承して新しいクラスを生成
class LSTMClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size):
        #superで親クラスのコンストラクタの呼び出し
        super().__init__()
        #単語ベクトルに変換
        self.word_embeddings = nn.Embedding(v_size, e_size)
        #LSTM（1層）
        self.lstm = nn.LSTM(e_size, h_size, num_layers = 1)
        #softmax関数に入れる前に全結合
        self.out = nn.Linear(h_size, c_size)
        self.softmax = nn.Softmax(dim=1)
        
    #順伝播    
    def forward(self, sentence):
        #入力された文章の単語をID番号に変換したものをテンソルへ
        sentence_matrix = torch.as_tensor(sent_to_ids(sentence), dtype=torch.int64)
        #単語ベクトルに変換
        embeds = self.word_embeddings(sentence_matrix)
        #print(embeds.size())
        
        #LSTMに入れるために次元数の調整（バッチサイズは1）
        #input of shape (seq_len, batch, input_size)
        reshape_matrix = embeds.view(len(sentence_matrix), 1, e_size)
        #print(reshape_matrix.size())
        
        #Outputs: output, (h_n, c_n)
        #h_n of shape (num_layers * num_directions, batch, hidden_size)
        #今回はカテゴリを最後に分類する問題なので、h_tを取得
        _, lstm_out = self.lstm(reshape_matrix)
        #print(lstm_out)
        
        tag_space = self.out(lstm_out[0].view(-1, h_size))
        #print(tag_space)
        #softmax関数
        tag_scores = self.softmax(tag_space)
        return tag_scores
    
model = LSTMClassifier(v_size, e_size, h_size, c_size)
model.forward('Fed official says weak data caused by weather, should not slow taper')#b


# In[386]:

model.forward('Love & Hip Hop star Benzino shot during mother\'s funeral procession')#e


# In[387]:

model.forward('Google Is Reportedly Set To Carve Up Its Failed Social Network Google+') #t


# In[388]:

model.forward('Dating via smartphone apps carry higher infection risk') #m


# ### 82. 確率的勾配降下法による学習
# 確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，問題81で構築したモデルを学習せよ．訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，適当な基準（例えば10エポックなど）で終了させよ．

# In[389]:

#カテゴリのラベルを変更
#ビジネス→0, 科学技術→1, エンターテイメント→2, 健康→3
#訓練データ
train_df.loc[train_df['CATEGORY'] == 'b', 'CATEGORY'] = 0
train_df.loc[train_df['CATEGORY'] == 't', 'CATEGORY'] = 1
train_df.loc[train_df['CATEGORY'] == 'e', 'CATEGORY'] = 2
train_df.loc[train_df['CATEGORY'] == 'm', 'CATEGORY'] = 3

#評価データ
test_df.loc[test_df['CATEGORY'] == 'b', 'CATEGORY'] = 0
test_df.loc[test_df['CATEGORY'] == 't', 'CATEGORY'] = 1
test_df.loc[test_df['CATEGORY'] == 'e', 'CATEGORY'] = 2
test_df.loc[test_df['CATEGORY'] == 'm', 'CATEGORY'] = 3


# In[390]:

#中身の確認
train_df


# In[391]:

test_df


# In[393]:

import torch.optim as optim

v_size = len(word_dict)
e_size = 300
h_size = 50
c_size = 4

#nn.Moduleを継承して新しいクラスを生成
class LSTMClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size):
        #superで親クラスのコンストラクタの呼び出し
        super().__init__()
        #単語ベクトルに変換 #padding_idxでパディングされた部分の埋め込みをすべて0
        self.word_embeddings = nn.Embedding(v_size, e_size, padding_idx=0)
        #LSTM（1層）
        self.lstm = nn.LSTM(e_size, h_size, num_layers = 1)
        #softmax関数に入れる前に全結合
        self.out = nn.Linear(h_size, c_size)
        self.softmax = nn.Softmax(dim=1)
        
    #順伝播    
    def forward(self, sentence):
        #入力された文章の単語をID番号に変換したものをテンソルへ
        sentence_matrix = torch.as_tensor(sent_to_ids(sentence), dtype=torch.int64)
        #単語ベクトルに変換
        embeds = self.word_embeddings(sentence_matrix)
        
        #LSTMに入れるために次元数の調整（バッチサイズは1）
        reshape_matrix = embeds.view(len(sentence_matrix), 1, e_size)
        
        #今回はカテゴリを最後に分類する問題なので、h_tを取得
        _, lstm_out = self.lstm(reshape_matrix)
        
        tag_space = self.out(lstm_out[0].view(-1, h_size))
        return tag_space

#モデル宣言
model = LSTMClassifier(v_size, e_size, h_size, c_size)
#損失関数（クロスエントロピー）
loss_function = nn.CrossEntropyLoss()
#SGDで最適化（学習係数0.01）
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    train_all_loss = 0
    test_all_loss = 0
    train_correct_num = 0
    test_correct_num = 0
    #訓練データによる学習
    for i in range(len(train_df)):
        # モデルが持ってる勾配の情報をリセット
        model.zero_grad()
        try:
            #順伝播の結果を受け取る
            out = model.forward(train_df['TITLE'][i])
            #カテゴリをテンソル化
            label = train_df['CATEGORY'][i]
            answer = torch.as_tensor([train_df['CATEGORY'][i]], dtype=torch.int64)
            #lossを計算する
            loss = loss_function(out, answer)
            #勾配をセット
            loss.backward()
            #逆伝播でパラメータ更新
            optimizer.step()
            train_all_loss += loss.item()
            #訓練データでの予測
            _, predict = torch.max(out, 1)
            #正解のとき
            if predict == answer:
                train_correct_num += 1
        #単語ベクトルに直せない文章を弾く
        except RuntimeError:
            pass
        
    #評価データでの予測   
    for i in range(len(test_df)):
        #順伝播の結果を受け取る
        out = model.forward(test_df['TITLE'][i])
        #カテゴリをテンソル化
        answer = torch.as_tensor([test_df['CATEGORY'][i]], dtype=torch.int64)
        #lossを計算する
        loss = loss_function(out, answer)
        test_all_loss += loss.item()
        #予測
        _, predict = torch.max(out, 1)
        #正解のとき
        if predict == answer:
            test_correct_num += 1
        
        train_accuracy = train_correct_num / len(train_df)
        test_accuracy = test_correct_num / len(test_df)
    print("epoch", epoch)
    print("訓練データ：", "損失", train_all_loss, "正解率", train_accuracy, sep="\t")
    print("評価データ：", "損失", test_all_loss, "正解率", test_accuracy, sep="\t", end="\n")


# ### 83. ミニバッチ化・GPU上での学習
# 問題82のコードを改変し，B事例ごとに損失・勾配を計算して学習を行えるようにせよ（Bの値は適当に選べ）．また，GPU上で学習を実行せよ．

# In[394]:

#ミニバッチを作る関数（訓練データからランダムに抽出）
def mini_batch(dataframe, b_size):
    shuffle_df = dataframe.sample(n = b_size) #nで行数指定
    category = shuffle_df['CATEGORY'].values
    title = shuffle_df['TITLE'].values
    return category, title

print(mini_batch(train_df, 250))


# In[553]:

import torch.optim as optim
import numpy as np

v_size = len(word_dict)
e_size = 300
h_size = 50
c_size = 4
b_size = 128 #バッチサイズ
b_iter = 100 #バッチ回数

#nn.Moduleを継承して新しいクラスを生成
class LSTMClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, b_size):
        #superで親クラスのコンストラクタの呼び出し
        super().__init__()
        #単語ベクトルに変換
        self.word_embeddings = nn.Embedding(v_size, e_size, padding_idx=0)
        #LSTM（1層）#batch_firstのとき、(batch, seq, feature) 
        self.lstm = nn.LSTM(e_size, h_size, num_layers = 1, batch_first=True)
        #softmax関数に入れる前に全結合
        self.out = nn.Linear(h_size, c_size)
        self.softmax = nn.Softmax(dim=1)
        
    #順伝播 
    def forward(self, sentence, b_size):
        for i in range(b_size):
            #入力された文章の単語をID番号に変換したものをテンソルへ
            sentence_matrix = torch.as_tensor(sent_to_ids(sentence[i]), dtype=torch.int64)
            #単語ベクトルに変換
            x = self.word_embeddings(sentence_matrix)
            if i != 0:
                embeds = torch.cat((embeds, x), 0)
            else:
                embeds = x

        #LSTMに入れるために次元数の調整
        reshape_matrix = embeds.view(b_size, max_len, e_size)
        
        #今回はカテゴリを最後に分類する問題なので、h_tを取得
        _, lstm_out = self.lstm(reshape_matrix)
        
        tag_space = self.out(lstm_out[0].view(-1, h_size))
        return tag_space

#モデル宣言
model = LSTMClassifier(v_size, e_size, h_size, c_size, b_size)
#損失関数（クロスエントロピー）
loss_function = nn.CrossEntropyLoss()
#SGDで最適化（学習係数0.01）
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    train_all_loss = 0
    test_all_loss = 0
    train_correct_num = 0
    test_correct_num = 0
    #訓練データによる学習
    for i in range(b_iter):
        #ミニバッチの作成
        train_category, train_title = mini_batch(train_df, b_size)
        
        # モデルが持ってる勾配の情報をリセット
        model.zero_grad()
        #順伝播の結果を受け取る
        out = model.forward(train_title, b_size)
        #カテゴリをテンソル化
        answer = torch.from_numpy(train_category.astype(np.int64)).clone()
        #lossを計算する
        loss = loss_function(out, answer)
        #勾配をセット
        loss.backward()
        #逆伝播でパラメータ更新
        optimizer.step()
        train_all_loss += loss.item()
        #訓練データでの予測
        _, predict = torch.max(out, 1)
        #正解のとき
        for j in range(b_size):
            if predict[j] == answer[j]:
                train_correct_num += 1 
            
        
    #評価データでの予測   
    for i in range(b_iter):
        #ミニバッチの作成
        test_category, test_title = mini_batch(test_df, b_size)
        #順伝播の結果を受け取る
        out = model.forward(test_title, b_size)
        #カテゴリをテンソル化
        answer = torch.from_numpy(test_category.astype(np.int64)).clone()
        #lossを計算する
        loss = loss_function(out, answer)
        test_all_loss += loss.item()
        #予測
        _, predict = torch.max(out, 1)
        #正解のとき
        for j in range(b_size):
            if predict[j] == answer[j]:
                test_correct_num += 1 
        
        train_accuracy = train_correct_num / (b_iter * b_size)
        test_accuracy = test_correct_num / (b_iter * b_size)
    print("epoch", epoch)
    print("訓練データ：", "損失", train_all_loss, "正解率", train_accuracy, sep="\t")
    print("評価データ：", "損失", test_all_loss, "正解率", test_accuracy, sep="\t", end="\n")


# ### 84. 単語ベクトルの導入
# 事前学習済みの単語ベクトル（例えば，Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル）で単語埋め込みemb(x)を初期化し，学習せよ．

# In[420]:

from gensim.models import KeyedVectors

#学習済の重み行列の読み込み
vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


# In[546]:

import torch

# テキスト内の単語ベクトルを返す関数
def Embedding(text):
    words_list = []
    s = re.sub(r"[!\"#\$%&'\(\)-=~\^\|@`\{\}\[\]:\*;\+,<\.>?/\d]", '', text) #記号の削除
    words = word_tokenaize(s)
    for word in words:
        norm = word_normalization(word)
        if not norm in stopword_list and norm in word_dict.keys() and norm in vectors: #ストップワードを削除 #単語がword_dictに存在するとき
            words_list.append(vectors[norm])
    #パディング
    for j in range(max_len - len(words_list)):
        words_list.insert(0, np.zeros(300))
    #サイズ確認    
    #print( torch.as_tensor(words_list, dtype=torch.float).size())
    return torch.as_tensor(words_list, dtype=torch.float)


# In[547]:

print(Embedding('Fed official says weak data caused by weather, should not slow taper'))


# In[555]:

import torch.optim as optim
import numpy as np

v_size = len(word_dict)
e_size = 300
h_size = 50
c_size = 4
b_size = 128 #バッチサイズ
b_iter = 100 #バッチ回数

#nn.Moduleを継承して新しいクラスを生成
class LSTMClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, b_size):
        #superで親クラスのコンストラクタの呼び出し
        super().__init__()
        #LSTM（1層）#batch_firstのとき、(batch, seq, feature) 
        self.lstm = nn.LSTM(e_size, h_size, num_layers = 1, batch_first=True)
        #softmax関数に入れる前に全結合
        self.out = nn.Linear(h_size, c_size)
        self.softmax = nn.Softmax(dim=1)
        
    #順伝播 
    def forward(self, sentence, b_size):
        for i in range(b_size):
            #入力された文章の単語を単語ベクトルに変換
            x = Embedding(sentence[i])
            if i != 0:
                embeds = torch.cat((embeds, x), 0)
            else:
                embeds = x
        
        #LSTMに入れるために次元数の調整
        reshape_matrix = embeds.view(b_size, max_len, e_size)
        
        #今回はカテゴリを最後に分類する問題なので、h_tを取得
        _, lstm_out = self.lstm(reshape_matrix)
        
        tag_space = self.out(lstm_out[0].view(-1, h_size))
        return tag_space

#モデル宣言
model = LSTMClassifier(v_size, e_size, h_size, c_size, b_size)
#損失関数（クロスエントロピー）
loss_function = nn.CrossEntropyLoss()
#SGDで最適化（学習係数0.01）
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    train_all_loss = 0
    test_all_loss = 0
    train_correct_num = 0
    test_correct_num = 0
    #訓練データによる学習
    for i in range(b_iter):
        #ミニバッチの作成
        train_category, train_title = mini_batch(train_df, b_size)
        
        # モデルが持ってる勾配の情報をリセット
        model.zero_grad()
        #順伝播の結果を受け取る
        out = model.forward(train_title, b_size)
        #カテゴリをテンソル化
        answer = torch.from_numpy(train_category.astype(np.int64)).clone()
        #lossを計算する
        loss = loss_function(out, answer)
        #勾配をセット
        loss.backward()
        #逆伝播でパラメータ更新
        optimizer.step()
        train_all_loss += loss.item()
        #訓練データでの予測
        _, predict = torch.max(out, 1)
        #正解のとき
        for j in range(b_size):
            if predict[j] == answer[j]:
                train_correct_num += 1 
            
        
    #評価データでの予測   
    for i in range(b_iter):
        #ミニバッチの作成
        test_category, test_title = mini_batch(test_df, b_size)
        #順伝播の結果を受け取る
        out = model.forward(test_title, b_size)
        #カテゴリをテンソル化
        answer = torch.from_numpy(test_category.astype(np.int64)).clone()
        #lossを計算する
        loss = loss_function(out, answer)
        test_all_loss += loss.item()
        #予測
        _, predict = torch.max(out, 1)
        #正解のとき
        for j in range(b_size):
            if predict[j] == answer[j]:
                test_correct_num += 1 
        
        train_accuracy = train_correct_num / (b_iter * b_size)
        test_accuracy = test_correct_num / (b_iter * b_size)
    print("epoch", epoch)
    print("訓練データ：", "損失", train_all_loss, "正解率", train_accuracy, sep="\t")
    print("評価データ：", "損失", test_all_loss, "正解率", test_accuracy, sep="\t", end="\n")


# In[556]:

#バッチ学習
import torch.optim as optim

v_size = len(word_dict)
e_size = 300
h_size = 50
c_size = 4

#nn.Moduleを継承して新しいクラスを生成
class LSTMClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size):
        #superで親クラスのコンストラクタの呼び出し
        super().__init__()
        #LSTM（1層）
        self.lstm = nn.LSTM(e_size, h_size, num_layers = 1)
        #softmax関数に入れる前に全結合
        self.out = nn.Linear(h_size, c_size)
        self.softmax = nn.Softmax(dim=1)
        
    #順伝播    
    def forward(self, sentence):
        #単語ベクトルに変換
        embeds = Embedding(sentence_matrix)
        
        #LSTMに入れるために次元数の調整（バッチサイズは1）
        reshape_matrix = embeds.view(len(sentence_matrix), 1, e_size)
        
        #今回はカテゴリを最後に分類する問題なので、h_tを取得
        _, lstm_out = self.lstm(reshape_matrix)
        
        tag_space = self.out(lstm_out[0].view(-1, h_size))
        return tag_space

#モデル宣言
model = LSTMClassifier(v_size, e_size, h_size, c_size)
#損失関数（クロスエントロピー）
loss_function = nn.CrossEntropyLoss()
#SGDで最適化（学習係数0.01）
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    train_all_loss = 0
    test_all_loss = 0
    train_correct_num = 0
    test_correct_num = 0
    #訓練データによる学習
    for i in range(len(train_df)):
        # モデルが持ってる勾配の情報をリセット
        model.zero_grad()
        try:
            #順伝播の結果を受け取る
            out = model.forward(train_df['TITLE'][i])
            #カテゴリをテンソル化
            label = train_df['CATEGORY'][i]
            answer = torch.as_tensor([train_df['CATEGORY'][i]], dtype=torch.int64)
            #lossを計算する
            loss = loss_function(out, answer)
            #勾配をセット
            loss.backward()
            #逆伝播でパラメータ更新
            optimizer.step()
            train_all_loss += loss.item()
            #訓練データでの予測
            _, predict = torch.max(out, 1)
            #正解のとき
            if predict == answer:
                train_correct_num += 1
        #単語ベクトルに直せない文章を弾く
        except RuntimeError:
            pass
        
    #評価データでの予測   
    for i in range(len(test_df)):
        #順伝播の結果を受け取る
        out = model.forward(test_df['TITLE'][i])
        #カテゴリをテンソル化
        answer = torch.as_tensor([test_df['CATEGORY'][i]], dtype=torch.int64)
        #lossを計算する
        loss = loss_function(out, answer)
        test_all_loss += loss.item()
        #予測
        _, predict = torch.max(out, 1)
        #正解のとき
        if predict == answer:
            test_correct_num += 1
        
        train_accuracy = train_correct_num / len(train_df)
        test_accuracy = test_correct_num / len(test_df)
    print("epoch", epoch)
    print("訓練データ：", "損失", train_all_loss, "正解率", train_accuracy, sep="\t")
    print("評価データ：", "損失", test_all_loss, "正解率", test_accuracy, sep="\t", end="\n")


# ### 85. 双方向RNN・多層化
# 順方向と逆方向のRNNの両方を用いて入力テキストをエンコードし，モデルを学習せよ．
# 
# h⃖ T+1=0
# 
# h⃖ t=RNN←−−−(emb(xt),h⃖ t+1)
# 
# y=softmax(W(yh)[h→T;h⃖ 1]+b(y))
# 
# 
# ただし，h→t∈ℝdh,h⃖ t∈ℝdhはそれぞれ，順方向および逆方向のRNNで求めた時刻tの隠れ状態ベクトル，RNN←−−−(x,h)は入力xと次時刻の隠れ状態hから前状態を計算するRNNユニット，W(yh)∈ℝL×2dhは隠れ状態ベクトルからカテゴリを予測するための行列，b(y)∈ℝLはバイアス項である．また，[a;b]はベクトルaとbの連結を表す。
# 
# さらに，双方向RNNを多層化して実験せよ．

# In[535]:

import torch.optim as optim
import numpy as np

v_size = len(word_dict)
e_size = 300
h_size = 50
c_size = 4
b_size = 128 #バッチサイズ
b_iter = 100 #バッチ回数

#nn.Moduleを継承して新しいクラスを生成
class LSTMClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, b_size):
        #superで親クラスのコンストラクタの呼び出し
        super().__init__()
        #単語ベクトルに変換
        self.word_embeddings = nn.Embedding(v_size, e_size, padding_idx=0)
        #LSTM（今回は2層）#batch_firstのとき、(batch, seq, feature) #bidirectionalで双方向LSTMに
        self.lstm = nn.LSTM(e_size, h_size, num_layers = 2, batch_first=True, bidirectional=True)
        #softmax関数に入れる前に全結合
        self.out = nn.Linear(h_size, c_size)
        self.softmax = nn.Softmax(dim=1)
        
    #順伝播 
    def forward(self, sentence, b_size):
        for i in range(b_size):
            #入力された文章の単語をID番号に変換したものをテンソルへ
            sentence_matrix = torch.as_tensor(sent_to_ids(sentence[i]), dtype=torch.int64)
            #単語ベクトルに変換
            x = self.word_embeddings(sentence_matrix)
            if i != 0:
                embeds = torch.cat((embeds, x), 0)
            else:
                embeds = x
        
        #LSTMに入れるために次元数の調整
        reshape_matrix = embeds.view(b_size, max_len, e_size)
        
        #今回はカテゴリを最後に分類する問題なので、h_tを取得
        _, lstm_out = self.lstm(reshape_matrix)
        
        tag_space = self.out(lstm_out[0].view(-1, h_size))
        return tag_space

#モデル宣言
model = LSTMClassifier(v_size, e_size, h_size, c_size, b_size)
#損失関数（クロスエントロピー）
loss_function = nn.CrossEntropyLoss()
#SGDで最適化（学習係数0.01）
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    train_all_loss = 0
    test_all_loss = 0
    train_correct_num = 0
    test_correct_num = 0
    #訓練データによる学習
    for i in range(b_iter):
        #ミニバッチの作成
        train_category, train_title = mini_batch(train_df, b_size)
        
        # モデルが持ってる勾配の情報をリセット
        model.zero_grad()
        #順伝播の結果を受け取る
        out = model.forward(train_title, b_size)
        #カテゴリをテンソル化
        answer = torch.from_numpy(train_category.astype(np.int64)).clone()
        #lossを計算する
        loss = loss_function(out, answer)
        #勾配をセット
        loss.backward()
        #逆伝播でパラメータ更新
        optimizer.step()
        train_all_loss += loss.item()
        #訓練データでの予測
        _, predict = torch.max(out, 1)
        #正解のとき
        for j in range(b_size):
            if predict[j] == answer[j]:
                train_correct_num += 1 
            
        
    #評価データでの予測   
    for i in range(b_iter):
        #ミニバッチの作成
        test_category, test_title = mini_batch(test_df, b_size)
        #順伝播の結果を受け取る
        out = model.forward(test_title, b_size)
        #カテゴリをテンソル化
        answer = torch.from_numpy(test_category.astype(np.int64)).clone()
        #lossを計算する
        loss = loss_function(out, answer)
        test_all_loss += loss.item()
        #予測
        _, predict = torch.max(out, 1)
        #正解のとき
        for j in range(b_size):
            if predict[j] == answer[j]:
                test_correct_num += 1 
        
        train_accuracy = train_correct_num / (b_iter * b_size)
        test_accuracy = test_correct_num / (b_iter * b_size)
    print("epoch", epoch)
    print("訓練データ：", "損失", train_all_loss, "正解率", train_accuracy, sep="\t")
    print("評価データ：", "損失", test_all_loss, "正解率", test_accuracy, sep="\t", end="\n")





