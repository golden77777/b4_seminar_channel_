
# coding: utf-8

# In[2]:

import warnings
warnings.simplefilter('ignore')


# # 70. 単語ベクトルの和による特徴量

# In[129]:

# train,test,validデータの読み込み
import pandas as pd

train = pd.read_table("train.txt")
test = pd.read_table("test.txt")
valid = pd.read_table('valid.txt')

#train.head()

def category_titile(data):
    data_title = data[["category", "title_text"]]
    data_title.loc[data_title['category'] == "b", 'category'] = 0
    data_title.loc[data_title['category'] == "t", 'category'] = 1
    data_title.loc[data_title['category'] == "e", 'category'] = 2
    data_title.loc[data_title['category'] == "m", 'category'] = 3
    
    return data_title

train = category_titile(train)
test = category_titile(test)
valid = category_titile(valid)

train.head()


# In[130]:

# 単語ベクトルのデータの読み込み
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[131]:

# 記事の内容を単語ベクトルに変換
# 今回は説明変数に記事で使われている単語の単語ベクトル(300次元)の平均をとる。
import numpy as np

vocab = list(model.vocab.keys())

def data_vector(data):
    data_vector = data.copy()
    data_vector["word"] = 0
    for i in range(len(data_vector)):
        words = data_vector["title_text"][i].split(" ")
        data_vector["word"][i] = words
    vectors_list = []
    for i in range(len(data_vector)):
        vector = np.zeros(300)
        count = 0
        for word in data_vector["word"][i]:
            if word not in vocab:
                continue
            vector += model[word]
            count += 1
        vectors_list.append(vector/count)
    vectors_df = pd.DataFrame(vectors_list)
    data_vector = pd.concat([data_vector, vectors_df],axis=1)
    return data_vector


# In[132]:

# 記事の内容を単語ベクトルに変換
# 今回は説明変数に記事で使われている単語の単語ベクトル(300次元)の平均をとる。
import numpy as np

vocab = list(model.vocab.keys())

def data_vector(data):
    data_vector = data.copy()
    data_vector["word"] = 0
    for i in range(len(data_vector)):
        words = data_vector["title_text"][i].split(" ")
        data_vector["word"][i] = words
    vectors_list = []
    for i in range(len(data_vector)):
        vector = np.zeros(300)
        for word in data_vector["word"][i]:
            if word not in vocab:
                continue
            vector += model[word]
        vectors_list.append(vector/len(data_vector["word"][i]))
    vectors_df = pd.DataFrame(vectors_list)
    data_vector = pd.concat([data_vector, vectors_df],axis=1)
    return data_vector


# In[133]:

train = data_vector(train)
train.head()


# In[134]:

test = data_vector(test)


# In[135]:

valid = data_vector(valid)


# In[150]:

X_train = train.drop(["category", "title_text", "word"], axis=1)
y_train = train["category"]
X_test = test.drop(["category", "title_text", "word"], axis=1)
y_test = test["category"]
X_valid = valid.drop(["category", "title_text", "word"], axis=1)
y_valid = valid["category"]


# In[151]:

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_valid.shape)
print(y_valid.shape)


# In[152]:

X_train.to_csv("X_train.csv")
y_train.to_csv("y_train.csv")
X_test.to_csv("X_test.csv")
y_test.to_csv("y_test.csv")
X_valid.to_csv("X_valid.csv")
y_valid.to_csv("y_valid.csv")


# In[153]:

X_train.head()


# In[154]:

y_train.head()


# # 71. 単層ニューラルネットワークによる予測

# In[155]:

import torch

X_train = X_train.astype(np.float32)
X_train = np.array(X_train)
X_train = torch.from_numpy(X_train)
y_train = torch.tensor(y_train)

X_test = X_test.astype(np.float32)
X_test = np.array(X_test)
X_test = torch.from_numpy(X_test)
y_test = torch.tensor(y_test)

X_valid = X_valid.astype(np.float32)
X_valid = np.array(X_valid)
X_valid = torch.from_numpy(X_valid)
y_valid = torch.tensor(y_valid)


# In[156]:

X_train


# In[157]:

y_train


# In[158]:

import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, d, l):
        super().__init__()
        self.fc = nn.Linear(d, l, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
    
    def forward(self, x):
        x = self.fc(x)
        return x


# In[159]:

perceptron = Perceptron(300, 4)


# In[160]:

x = perceptron(X_train[0])
x = torch.softmax(x, dim=-1)
x


# In[161]:

x = perceptron(X_train[0:4])
x = torch.softmax(x, dim=-1)
x


# # 72. 損失と勾配の計算

# In[162]:

criterion = nn.CrossEntropyLoss()
y_pred = perceptron(X_train[:1])
y_ture = y_train[:1]
loss = criterion(y_pred, y_ture)
perceptron.zero_grad()
loss.backward()
print('損失 :', loss.item())
print('勾配')
print(perceptron.fc.weight.grad)


# In[163]:

y_pred = perceptron(X_train[:4])
y_ture = y_train[:4]
loss = criterion(y_pred, y_ture)
perceptron.zero_grad()
loss.backward()
print('損失 :', loss.item())
print('勾配')
print(perceptron.fc.weight.grad)


# In[164]:

y_pred = perceptron(X_train)
y_ture = y_train
loss = criterion(y_pred, y_ture)
perceptron.zero_grad()
loss.backward()
print('損失 :', loss.item())
print('勾配')
print(perceptron.fc.weight.grad)


# # 73. 確率的勾配降下法による学習

# In[168]:

import torch.optim as optim
perceptron = Perceptron(300, 4)
optimizer = optim.SGD(perceptron.parameters(), lr=0.001, momentum=0.9)


for epoch in range(10000):
    outputs = perceptron(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print('Epoch [{}/{}], Loss {:.4f}'.format(epoch+1, 10000, loss.item()))


# # 74. 正解率の計測

# In[169]:

def argmax(pred):
    pred_list = []
    for i in range(len(pred)):
        pred_list.append(np.argmax(pred[i]))
    return pred_list

def accuracy(true, pred):
    return np.mean([t == p for t, p in zip(true, pred)])


# In[170]:

y_train_pred = argmax(perceptron(X_train).data.numpy())
y_train.data.numpy()

print("train正答率: {}".format(accuracy(y_train, y_train_pred)))

y_valid_pred = argmax(perceptron(X_valid).data.numpy())
y_valid.data.numpy()

print("valid正答率: {}".format(accuracy(y_valid, y_valid_pred)))

y_test_pred = argmax(perceptron(X_test).data.numpy())
y_test.data.numpy()

print("test正答率: {}".format(accuracy(y_test, y_test_pred)))


# # 75. 損失と正解率のプロット

# In[177]:

import torch.optim as optim

perceptron = Perceptron(300, 4)
optimizer = optim.SGD(perceptron.parameters(), lr=0.001, momentum=0.9)

train_acc = []
train_loss = []
valid_acc = []
valid_loss = []
epoch_list = []

for epoch in range(10000):
    outputs = perceptron(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print('Epoch [{}/{}], Loss {:.4f}'.format(epoch+1, 10000, loss.item()))
        y_train_pred = argmax(perceptron(X_train).data.numpy())
        y_train.data.numpy()
        train_acc.append(accuracy(y_train, y_train_pred))
        y_valid_pred = argmax(perceptron(X_valid).data.numpy())
        y_valid.data.numpy()
        valid_acc.append(accuracy(y_valid, y_valid_pred))
        train_output = perceptron(X_train)
        train_loss.append(criterion(train_output, y_train))
        valid_output = perceptron(X_valid)
        valid_loss.append(criterion(valid_output, y_valid))
        epoch_list.append(epoch+1)


# In[178]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.plot(epoch_list, train_acc, label="train")
plt.plot(epoch_list, valid_acc, label="valid")

plt.title("acc")
plt.legend()

plt.show()


# In[179]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.plot(epoch_list, train_loss, label="train")
plt.plot(epoch_list, valid_loss, label="valid")

plt.title("loss")
plt.legend()

plt.show()


# # 76. チェックポイント

# In[182]:

import torch.optim as optim
perceptron = Perceptron(300, 4)
optimizer = optim.SGD(perceptron.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10000):
    outputs = perceptron(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print('Epoch [{}/{}], Loss {:.4f}'.format(epoch+1, 10000, loss.item()))
        torch.save({'epoch' : (epoch+1)//10, 'optimizer': optimizer}, f'trainer_states{(epoch+1)//10}.pt')
        torch.save(perceptron.state_dict(), f'checkpoint{(epoch+1)//10}.pt')


# In[183]:

model_path = 'checkpoint1.pt'
perceptron.load_state_dict(torch.load(model_path))
for param in perceptron.parameters():
    print(param)


# # 77. ミニバッチ化

# In[188]:

import torch.optim as optim
from torch.utils.data import DataLoader
perceptron = Perceptron(300, 4)
optimizer = optim.SGD(perceptron.parameters(), lr=0.001, momentum=0.9)

class DataSet():
    def __init__(self):
        self.X = X_train
        self.y = y_train
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

dataset = DataSet()
dataloader = DataLoader(dataset, batch_size=5)

for epoch in range(100):
    
    for data in dataloader:
        outputs = perceptron(data[0])
        loss = criterion(outputs, data[1])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss {:.4f}'.format(epoch+1, 100, loss.item()))


# In[189]:

y_train_pred = argmax(perceptron(X_train).data.numpy())
y_train.data.numpy()

print("train正答率: {}".format(accuracy(y_train, y_train_pred)))

y_valid_pred = argmax(perceptron(X_valid).data.numpy())
y_valid.data.numpy()

print("valid正答率: {}".format(accuracy(y_valid, y_valid_pred)))

y_test_pred = argmax(perceptron(X_test).data.numpy())
y_test.data.numpy()

print("test正答率: {}".format(accuracy(y_test, y_test_pred)))


# In[198]:

import torch.optim as optim
from torch.utils.data import DataLoader
import time

class DataSet():
    def __init__(self):
        self.X = X_train
        self.y = y_train
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

dataset = DataSet()
time_list = []
batch_list = [1, 2, 4, 8]

for batch in batch_list:
    perceptron = Perceptron(300, 4)
    optimizer = optim.SGD(perceptron.parameters(), lr=0.001, momentum=0.9)
    dataloader = DataLoader(dataset, batch_size=batch)
    print('Batch Size : {}'.format(batch))
    start = time.time()
    for epoch in range(100):
        
        for data in dataloader:
            outputs = perceptron(data[0])
            loss = criterion(outputs, data[1])
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss {:.4f}'.format(epoch+1, 100, loss.item()))
    end = time.time()
    time_list.append(end-start)
for i in range(len(time_list)):
    print('Batch : {}, Time : {}'.format(batch_list[i], time_list[i]/100))


# In[199]:

import torch.optim as optim
from torch.utils.data import DataLoader
import time

class DataSet():
    def __init__(self):
        self.X = X_train
        self.y = y_train
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

dataset = DataSet()
time_list = []
batch_list = [1]

for batch in batch_list:
    perceptron = Perceptron(300, 4)
    optimizer = optim.SGD(perceptron.parameters(), lr=0.001, momentum=0.9)
    dataloader = DataLoader(dataset, batch_size=batch)
    print('Batch Size : {}'.format(batch))
    for epoch in range(100):
        
        for data in dataloader:
            outputs = perceptron(data[0])
            loss = criterion(outputs, data[1])
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss {:.4f}'.format(epoch+1, 100, loss.item()))
    
y_train_pred = argmax(perceptron(X_train).data.numpy())
y_train.data.numpy()

print("train正答率: {}".format(accuracy(y_train, y_train_pred)))

y_valid_pred = argmax(perceptron(X_valid).data.numpy())
y_valid.data.numpy()

print("valid正答率: {}".format(accuracy(y_valid, y_valid_pred)))

y_test_pred = argmax(perceptron(X_test).data.numpy())
y_test.data.numpy()

print("test正答率: {}".format(accuracy(y_test, y_test_pred)))


# # 78. GPU上での学習

# In[ ]:

import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, d, l):
        super().__init__()
        self.fc = nn.Linear(d, l, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
    
    def forward(self, x):
        x = self.fc(x)
        return x

import torch.optim as optim
from torch.utils.data import DataLoader
import time

device = 'cuda'
device = torch.device('cuda')

class DataSet():
    def __init__(self):
        self.X = X_train.to('cuda')
        self.y = y_train.to('cuda')
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

dataset = DataSet()
time_list = []
batch_list = [1, 2, 4, 8]

for batch in batch_list:
    perceptron = Perceptron(300, 4)
    perceptron = perceptron.to(device)
    dataloader = DataLoader(dataset, batch_size=batch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(perceptron.parameters(), lr=0.001, momentum=0.9)
    print('Batch Size : {}'.format(batch))
    start = time.time()
    for epoch in range(10):
        
        for data in dataloader:
            outputs = perceptron(data[0])
            loss = criterion(outputs, data[1])
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        if (epoch+1) % 1 == 0:
            print('Epoch [{}/{}], Loss {:.4f}'.format(epoch+1, 10, loss.item()))
    end = time.time()
    time_list.append(end-start)
for i in range(len(time_list)):
    print('Batch : {}, Time : {}'.format(batch_list[i], time_list[i]/10))


# # 79. 多層ニューラルネットワーク

# In[ ]:

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random

class Perceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        #p = random.random()
        x = F.dropout(x, 0.5, training=self.training)
        x = self.bn1(x)
        x = self.fc2(x)
        return x

import time

device = 'cuda'

class DataSet():
    def __init__(self):
        self.X = X_train.to('cuda')
        self.y = y_train.to('cuda')
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

dataset = DataSet()
time_list = []
batch_list = [10]

for batch in batch_list:
    perceptron = Perceptron(300, 30, 4)
    perceptron = perceptron.to(device)
    perceptron.train()
    dataloader = DataLoader(dataset, batch_size=batch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(perceptron.parameters(), lr=0.001)
    print('Batch Size : {}'.format(batch))
    start = time.time()
    for epoch in range(100):
        
        for data in dataloader:
            outputs = perceptron(data[0])
            loss = criterion(outputs, data[1])
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss {:.4f}'.format(epoch+1, 100, loss.item()))
    end = time.time()
    time_list.append(end-start)

