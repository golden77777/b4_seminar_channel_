
# coding: utf-8

# In[1]:

import warnings
warnings.simplefilter('ignore')


# # 50. データの入手・整形

# In[2]:

import pandas as pd

# csvファイルの読み込み
two_page_sessions_columns_names = ["story", "hostname", "category", "url"]
two_page_sessions = pd.read_table("NewsAggregatorDataset/2pageSessions.csv", names=two_page_sessions_columns_names)
news_corpora_columns_names = ["id", "title", "url", "publisher", "category", "story", "hostname", "timestamp"]
news_corpora = pd.read_table("NewsAggregatorDataset/newsCorpora.csv", names=news_corpora_columns_names)


# In[3]:

print(len(two_page_sessions))
two_page_sessions.head()


# In[4]:

two_page_sessions.isnull().sum()


# In[5]:

print(len(news_corpora))
news_corpora.head()


# In[6]:

news_corpora.isnull().sum()


# In[7]:

#データの結合(storyをkeyに結合で、rightジョイン)
#join_data = pd.merge(news_corpora, two_page_sessions, on=["story", "category"], how="right")
#print(len(join_data))
#join_data.head()
join_data = news_corpora


# In[8]:

join_data.isnull().sum()


# In[9]:

# 見やすいように順番を入れ替える
#join_data = join_data[["id", "story", "category", "title", "publisher", "timestamp", "url_x", "hostname_x", "url_y", "hostname_y"]]
join_data = join_data[["id", "category","title", "publisher", "url", "story", "hostname", "timestamp"]]
join_data.head()


# In[10]:

# publisherで”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”のみを抽出
publisher_data = pd.DataFrame()
for publisher in ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]:
    publisher_data = pd.concat([publisher_data, join_data.loc[join_data["publisher"] ==publisher]])
print(len(publisher_data))
publisher_data.head()


# In[11]:

# それぞれの出版元があるか確認
publisher_data["publisher"].value_counts()


# In[12]:

# ランダムに並び替え
random_data = publisher_data.sample(frac=1, random_state=0).reset_index(drop=True)
random_data.head()


# In[13]:

# データの分割
random_data = random_data.rename(columns={"title":"title_text"})
train_data = random_data[:round(len(random_data)*0.8)]
valid_data = random_data[round(len(random_data)*0.8):round(len(random_data)*0.9)]
test_data = random_data[round(len(random_data)*0.9):]
print(len(train_data))
print(len(valid_data))
print(len(test_data))


# In[14]:

# ファイル出力
train_data.to_csv("train.txt", sep="\t")
valid_data.to_csv("valid.txt", sep="\t")
test_data.to_csv("test.txt", sep="\t")


# # 51. 特徴量抽出

# In[15]:

# タイトルのみに注目
train_data_feature = train_data[["category", "title_text"]]

# いらない記号の削除
train_data_feature["title_text"] = train_data_feature["title_text"].replace('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠©。、？！｀＋￥％]', '', regex=True)


# In[16]:

# 形態素解析treetagger
import treetaggerwrapper

tagger = treetaggerwrapper.TreeTagger(TAGLANG="en")

train_data_feature["title_treetagger"] = 0
for i in range(len(train_data_feature)):
    train_data_feature["title_treetagger"][i] = tagger.TagText(train_data_feature["title_text"][i])

train_data_feature.head()


# In[17]:

train_data_feature["category"].unique()


# In[18]:

print(type(train_data_feature["title_treetagger"][0]))
print(train_data_feature["title_treetagger"][0][1])


# In[19]:

# 単語ごとにリストに分ける
for i in range(len(train_data_feature)):
    for j in range(len(train_data_feature["title_treetagger"][i])):
        train_data_feature["title_treetagger"][i][j] = train_data_feature["title_treetagger"][i][j].split("\t")

train_data_feature.head()


# In[20]:

train_data_feature.to_csv("train_data_feature.csv")


# In[21]:

# 単語のみのリストを作る
def make_word_list(data):
    data['word_list'] = 0
    # 必要な品詞リスト
    # 形容詞、副詞、名詞、動詞
    pos_list = ["AJ0", "AJC", "AJS", "AV0", "NN0", "NN1", "NN2", "NP0", "VBB", "VBD", "VBG", "VBI", "VBN", "VBZ", "VDB", "VDD", "VDG", "VDI", "VDN", "VDZ", "VHB", "VHD", "VHG", "VHI", "VHZ", "VM0", "VVB", "VVD", "VVG", "VVI", "VVN", "VVZ"] 
    
    for i in range(len(data)):
        word_list = []
        for word in data['title_treetagger'][i]:
            if word[1] in pos_list:
                word_list.append(word[2])
        data['word_list'][i] = word_list


# In[22]:

make_word_list(train_data_feature)
train_data_feature.head()


# In[23]:

# どの単語がどれくらい使われているかを調べる
def word_count(data):
    # 必要な品詞リスト
    # 形容詞、副詞、名詞、動詞
    pos_list = ["AJ0", "AJC", "AJS", "AV0", "NN0", "NN1", "NN2", "NP0", "VBB", "VBD", "VBG", "VBI", "VBN", "VBZ", "VDB", "VDD", "VDG", "VDI", "VDN", "VDZ", "VHB", "VHD", "VHG", "VHI", "VHZ", "VM0", "VVB", "VVD", "VVG", "VVI", "VVN", "VVZ"] 

    # カウントするための辞書
    words_dict = dict()
    for sentence in data['title_treetagger']:
        for word in sentence:
            if word[1] in pos_list:
                if word[2] not in words_dict.keys():
                    words_dict[word[2]] = 1
                else:
                    words_dict[word[2]] += 1
    words_dict = sorted(words_dict.items(), key=lambda x:x[1], reverse=True)
    keys =  [word[0] for word in words_dict]
    values =  [word[1] for word in words_dict]
    words_dict = {k : v for k,v in zip(keys, values)}
    return words_dict


# In[24]:

# どの単語がどれくらい使われているかを調べる
def word_count(data):
    # カウントするための辞書
    words_dict = dict()
    for word_list in data["word_list"]:
        for word in word_list:
            if word not in words_dict.keys():
                words_dict[word] = 1
            else:
                words_dict[word] += 1
    words_dict = sorted(words_dict.items(), key=lambda x:x[1], reverse=True)
    keys =  [word[0] for word in words_dict]
    values =  [word[1] for word in words_dict]
    words_dict = {k : v for k,v in zip(keys, values)}
    return words_dict


# In[25]:

all_words_count = word_count(train_data_feature)
#「ビジネス」「科学技術」「エンターテイメント」「健康」
bussiness_word_count = word_count(train_data_feature.loc[train_data_feature["category"]=="b"])
science_word_count = word_count(train_data_feature.loc[train_data_feature["category"]=="t"])
entertainment_word_count = word_count(train_data_feature.loc[train_data_feature["category"]=="e"])
health_word_count = word_count(train_data_feature.loc[train_data_feature["category"]=="m"])

print(len(all_words_count))
print(len(bussiness_word_count))
print(len(science_word_count))
print(len(entertainment_word_count))
print(len(health_word_count))


# In[26]:

# matplotlibで可視化
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import japanize_matplotlib

def bar(word_count):
    index = 50
    x = list(word_count.keys())[:index]
    y = list(word_count.values())[:index]
    
    x_position = np.arange(len(x))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x_position, y, tick_label=x)
    ax.set_xlabel('単語')
    ax.set_ylabel('出現回数')
    fig.show()
    print(x)


# In[27]:

# 全て
bar(all_words_count)


# In[28]:

# ビジネス
bar(bussiness_word_count)


# In[29]:

# 科学技術
bar(science_word_count)


# In[30]:

# エンターテインメント
bar(entertainment_word_count)


# In[31]:

# 健康
bar(health_word_count)


# In[32]:

# 全てに共通する単語は削除
def delete_words(word_count):
    delete_key = []
    delete_word_count = word_count.copy()
    for key, value in word_count.items():
        if key in bussiness_word_count.keys() and key in science_word_count.keys() and key in entertainment_word_count.keys() and key in health_word_count.keys():
                delete_key.append(key)
    for key in delete_key:
        del delete_word_count[key]
    return delete_word_count


# In[68]:

print(len(all_words_count))
print(len(delete_word_count))


# In[33]:

delete_word_count = delete_words(all_words_count)
delete_bussiness_word_count = delete_words(bussiness_word_count)
delete_science_word_count = delete_words(science_word_count)
delete_entertainment_word_count = delete_words(entertainment_word_count)
delete_health_word_count = delete_words(health_word_count)


# In[34]:

# ビジネス
bar(delete_bussiness_word_count)


# In[35]:

# 科学技術
bar(delete_science_word_count)


# In[36]:

# エンターテインメント
bar(delete_entertainment_word_count)


# In[37]:

# 健康
bar(delete_health_word_count)


# In[ ]:




# In[32]:

# 全てのデータにおいて、カウントする
def category_count(data):
    data["category_e_word_count"] = 0
    data["category_b_word_count"] = 0
    data["category_m_word_count"] = 0
    data["category_t_word_count"] = 0
    
    for i in range(len(data)):
        e_count = 0
        b_count = 0
        m_count = 0
        t_count = 0
        for word in data["word_list"][i]:
            if word in entertainment_word_count.keys():
                e_count+=1
            if word in bussiness_word_count.keys():
                b_count+=1
            if word in health_word_count.keys():
                m_count+=1
            if word in science_word_count.keys():
                t_count+=1
        data["category_e_word_count"][i] = e_count
        data["category_b_word_count"][i] = b_count
        data["category_m_word_count"][i] = m_count
        data["category_t_word_count"][i] = t_count


# In[33]:

category_count(train_data_feature)
train_data_feature.head()


# In[34]:

# test_data, valid_dataにも同じような処理を行う
# タイトルのみに注目
valid_data_feature = valid_data[["category", "title_text"]].reset_index(drop=True)
test_data_feature = test_data[["category", "title_text"]].reset_index(drop=True)

# いらない記号の削除
valid_data_feature["title_text"] = valid_data_feature["title_text"].replace('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠©。、？！｀＋￥％]', '', regex=True)
test_data_feature["title_text"] = test_data_feature["title_text"].replace('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠©。、？！｀＋￥％]', '', regex=True)


# In[35]:

# 形態素解析
tagger = treetaggerwrapper.TreeTagger(TAGLANG="en")

valid_data_feature["title_treetagger"] = 0
for i in range(len(valid_data_feature)):
    valid_data_feature["title_treetagger"][i] = tagger.TagText(valid_data_feature["title_text"][i])

test_data_feature["title_treetagger"] = 0
for i in range(len(test_data_feature)):
    test_data_feature["title_treetagger"][i] = tagger.TagText(test_data_feature["title_text"][i])


# In[36]:

valid_data_feature.head()


# In[37]:

# 単語ごとにリストに分ける
for i in range(len(valid_data_feature)):
    for j in range(len(valid_data_feature["title_treetagger"][i])):
        valid_data_feature["title_treetagger"][i][j] = valid_data_feature["title_treetagger"][i][j].split("\t")

# 単語ごとにリストに分ける
for i in range(len(test_data_feature)):
    for j in range(len(test_data_feature["title_treetagger"][i])):
        test_data_feature["title_treetagger"][i][j] = test_data_feature["title_treetagger"][i][j].split("\t")


# In[38]:

# 単語のリストを作る
make_word_list(valid_data_feature)
make_word_list(test_data_feature)

# categoryごとの使用単語回数を数える
category_count(valid_data_feature)
category_count(test_data_feature)


# In[39]:

valid_data_feature.head()


# In[40]:

test_data_feature.head()


# In[44]:

# 単語ごとのカラムを作る
word_set = set()
for k, v in all_words_count.items():
    if v != 1:
        word_set.add(k)


# In[45]:

#カラム名を単語の名前にして、そこに単語の出現回数をいれる
def make_columns(data):
    for column in word_set:
        data[column] = 0
    for i in range(len(data)):
        for j in range(len(data["word_list"][i])):
            if data["word_list"][i][j] in word_set:
                data[data["word_list"][i][j]][i] += 1


# In[46]:

# train_dataに適応
make_columns(train_data_feature)
train_data_feature.head()


# In[47]:

# test, validにも適応
make_columns(test_data_feature)
make_columns(valid_data_feature)


# In[48]:

test_data_feature.head()


# In[49]:

valid_data_feature.head()


# In[53]:

# ファイル出力
train_feature = train_data_feature.drop(['title_text', 'title_treetagger', 'word_list'], axis=1)
valid_feature = valid_data_feature.drop(['title_text', 'title_treetagger', 'word_list'], axis=1)
test_feature = test_data_feature.drop(['title_text', 'title_treetagger', 'word_list'], axis=1)

train_feature.to_csv("train.feature.txt", sep="\t")
valid_feature.to_csv("valid.feature.txt", sep="\t")
test_feature.to_csv("test.feature.txt", sep="\t")


# In[54]:

train_feature.head()


# # 52. 学習

# In[51]:

train_feature = train_data_feature.drop(['title_text', 'title_treetagger', 'word_list'], axis=1)
valid_feature = valid_data_feature.drop(['title_text', 'title_treetagger', 'word_list'], axis=1)
test_feature = test_data_feature.drop(['title_text', 'title_treetagger', 'word_list'], axis=1)


# In[64]:

# ロジスティック回帰
from sklearn.linear_model import LogisticRegression

# データの分割
X_train = train_feature.drop(columns="category", axis=1)
y_train = train_feature["category"]
 
# 重回帰クラスの初期化と学習
model = LogisticRegression(class_weight='balanced')
model.fit(X_train,y_train)


# # 53. 予測

# In[56]:

def predict_category(text):
    tagger = treetaggerwrapper.TreeTagger(TAGLANG="en")

    tagger = tagger.TagText(text)
    word_list = []
    
    for word in tagger:
        word_list.append(word.split("\t")[2])

    category_e_word_count = 0
    category_b_word_count = 0
    category_m_word_count = 0
    category_t_word_count = 0
    
    for word in word_list:
        if word in delete_entertainment_word_count.keys():
            category_e_word_count+=1
        if word in delete_bussiness_word_count.keys():
            category_b_word_count+=1
        if word in delete_health_word_count.keys():
            category_m_word_count+=1
        if word in delete_science_word_count.keys():
            category_t_word_count+=1
    
    word_df = pd.DataFrame({'category_e_word_count': category_e_word_count, 'category_b_word_count': category_b_word_count,'category_m_word_count': category_m_word_count, 'category_t_word_count': category_t_word_count}, index=['0',])
    
    for column in word_set:
        word_df[column] = 0
        
    for word in word_list:
        if word in word_set:
            word_df[word] += 1
    
    
    y_pred = model.predict(word_df)
    y_probs = model.predict_proba(word_df)

    if y_pred[0] == "e":
        y_category = "entertainment"
    elif y_pred[0] == "b":
        y_category ="business"
    elif y_pred[0] == "m":
        y_category = "health"
    elif y_pred[0] == "t":
        y_category ="science and technology"        
    
    print("カテゴリー : {}".format(y_category))
    print("予測確率(b,e,m,t) : {}".format(y_probs))


# In[57]:

predict_category("Facebook takes aim at Snapchat with new slingshot self destructing message app")


# In[58]:

print(test_data_feature["title_text"][104])
print(test_data_feature["category"][104])


# # 54. 正解率の計測

# In[65]:

# データの分割
X_test = test_feature.drop(columns="category", axis=1)
y_test = test_feature["category"]
X_valid = valid_feature.drop(columns="category", axis=1)
y_valid = valid_feature["category"]

# 予測
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)
y_valid_predict = model.predict(X_valid)

from sklearn.metrics import accuracy_score

print('train :正解率 = {}'.format(accuracy_score(y_train, y_train_predict)))
print('test  :正解率 = {}'.format(accuracy_score(y_test, y_test_predict)))
print('valid  :正解率 = {}'.format(accuracy_score(y_valid, y_valid_predict)))


# # 55. 混同行列の作成

# In[66]:

from sklearn.metrics import confusion_matrix

print('train :混同行列 = \n{}'.format(confusion_matrix(y_train, y_train_predict)))
print('test  :混同行列 = \n{}'.format(confusion_matrix(y_test, y_test_predict)))
print('valid  :混同行列 = \n{}'.format(confusion_matrix(y_valid, y_valid_predict)))


# # 56. 適合率，再現率，F1スコアの計測

# In[67]:

from sklearn.metrics import precision_score, recall_score, f1_score

print('train :適合率 micro = {}'.format(precision_score(y_train, y_train_predict, average='micro')))
print('train :適合率 macro = {}'.format(precision_score(y_train, y_train_predict, average='macro')))
print('test :適合率 micro = {}'.format(precision_score(y_test, y_test_predict, average='micro')))
print('test :適合率 macro = {}'.format(precision_score(y_test, y_test_predict, average='macro')))
print('valid :適合率 micro = {}'.format(precision_score(y_valid, y_valid_predict, average='micro')))
print('valid :適合率 macro = {}'.format(precision_score(y_valid, y_valid_predict, average='macro')))
print()
print('train :再現率 micro = {}'.format(recall_score(y_train, y_train_predict, average='micro')))
print('train :再現率 macro = {}'.format(recall_score(y_train, y_train_predict, average='macro')))
print('test :再現率 micro = {}'.format(recall_score(y_test, y_test_predict, average='micro')))
print('test :再現率 macro = {}'.format(recall_score(y_test, y_test_predict, average='macro')))
print('valid :再現率 micro = {}'.format(recall_score(y_valid, y_valid_predict, average='micro')))
print('valid :再現率 macro = {}'.format(recall_score(y_valid, y_valid_predict, average='macro')))
print()
print('train :F1スコア micro = {}'.format(f1_score(y_train, y_train_predict, average='micro')))
print('train :F1スコア macro = {}'.format(f1_score(y_train, y_train_predict, average='macro')))
print('test :F1スコア micro = {}'.format(f1_score(y_test, y_test_predict, average='micro')))
print('test :F1スコア macro = {}'.format(f1_score(y_test, y_test_predict, average='macro')))
print('valid :F1スコア micro = {}'.format(f1_score(y_valid, y_valid_predict, average='micro')))
print('valid :F1スコア macro = {}'.format(f1_score(y_valid, y_valid_predict, average='macro')))
    


# # 57. 特徴量の重みの確認

# In[68]:

import numpy as np
print("business")
for col in np.argsort(model.coef_[0])[:-6:-1]:
    print(" {}: {}".format(X_test.columns[col], model.coef_[0][col]))
print()
for col in np.argsort(model.coef_[0])[:5]:
    print(" {}: {}".format(X_test.columns[col], model.coef_[0][col]))
print()
print("entertainment")
for col in np.argsort(model.coef_[1])[:-6:-1]:
    print(" {}: {}".format(X_test.columns[col], model.coef_[1][col]))
print()
for col in np.argsort(model.coef_[1])[:5]:
    print(" {}: {}".format(X_test.columns[col], model.coef_[1][col]))
print()
print("health")
for col in np.argsort(model.coef_[2])[:-6:-1]:
    print(" {}: {}".format(X_test.columns[col], model.coef_[2][col]))
print()
for col in np.argsort(model.coef_[2])[:5]:
    print(" {}: {}".format(X_test.columns[col], model.coef_[2][col]))
print()
print("science and technology")
for col in np.argsort(model.coef_[3])[:-6:-1]:
    print(" {}: {}".format(X_test.columns[col], model.coef_[3][col]))
print()
for col in np.argsort(model.coef_[3])[:5]:
    print(" {}: {}".format(X_test.columns[col], model.coef_[3][col]))


# # 58. 正則化パラメータの変更

# In[57]:

import matplotlib.pyplot as plt
C = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
C_train_acc = []
C_test_acc = []
C_valid_acc = []
for c in C:
    model_C = LogisticRegression(C=c)
    model_C.fit(X_train,y_train)
    train_pred = model_C.predict(X_train)
    C_train_acc.append(accuracy_score(y_train, train_pred))
    test_pred = model_C.predict(X_test)
    C_test_acc.append(accuracy_score(y_test, test_pred)) 
    valid_pred = model_C.predict(X_valid)
    C_valid_acc.append(accuracy_score(y_valid, valid_pred))


# In[58]:

#グラフの描写
plt.plot(C, C_train_acc, label='train')
plt.plot(C, C_test_acc, label='test')
plt.plot(C, C_valid_acc, label='valid')
plt.xscale('log')
plt.legend()
plt.show()


# # 59. ハイパーパラメータの探索

# In[71]:

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

grid_search = {LogisticRegression(): 
               {"C": [10 ** i for i in range(-3, 1)],
                "penalty": ['l1', 'l2']},
DecisionTreeClassifier():{"criterion": ["gini", "entropy"],
                                       "splitter": ["best", "random"],
                                       "max_depth": [1, 3, 5, 7, 10],
                                       "min_samples_split": [2, 5, 11],
                                       "min_samples_leaf": [2, 5, 11]
                                      }

}

max_score = 0

#グリッドサーチ
for model, param in grid_search.items():
    model_cv = GridSearchCV(model, param)
    model_cv.fit(X_train, y_train)
    y_test_pred = model_cv.predict(X_test)
    score = accuracy_score(y_test, y_test_pred)

    if max_score < score:
        max_score = score
        best_param = model_cv.best_params_
        best_model = model.__class__.__name__

print("ベストスコア:{}".format(max_score))
print("モデル:{}".format(best_model))
print("パラメーター:{}".format(best_param))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



