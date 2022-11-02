from tkinter import font
import torch
from torch import nn
import torch.optim as optim
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# read
t = []
x = []
file = open('data.csv')
csvReader = csv.reader(file)
for row in csvReader:
  t.append(datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
  x.append(float(row[1]))

# constants
seq = 10
n = len(x) - seq
batch_size = 16
train_percent = 0.5
n_train = int(n * train_percent)
lr = 0.001
num_epochs = 10

# process
x = torch.FloatTensor(x)
timestamps = t[seq:]
labels = x[seq:]
features = torch.zeros((n, seq, 1))
for i in range(seq):
  features[:, i] = x[i: n + i].unsqueeze(1)
train_features, train_labels = features[:n_train], labels[:n_train]
train_dataset = data.TensorDataset(train_features, train_labels)
train_iter = data.DataLoader(train_dataset, batch_size, shuffle=True)

# model
class GRU_REG(nn.Module):
  def __init__(self, input_size, hidden_size, seq, output_size, num_layers=2):
    super(GRU_REG, self).__init__()
    self.gru = nn.GRU(input_size, hidden_size, num_layers)
    self.fc = nn.Linear(hidden_size * seq, output_size)
  def forward(self, x):
    o, _ = self.gru(x)
    o = torch.transpose(o, 1, 0)
    b, s, h = o.shape
    o = o.reshape(b, s*h)
    o = self.fc(o)
    return o

# train
model = GRU_REG(input_size=1, hidden_size=2, output_size=1, num_layers=2, seq=seq)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
hist = np.zeros(num_epochs)
model.train()
for epoch in range(num_epochs):
  epoch_loss = 0
  for num, (X, y) in enumerate(train_iter):
    X = torch.transpose(X, 1, 0)
    y = y.reshape(-1, 1)
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  with torch.no_grad():
    epoch_loss += loss
  print(f'epoch {epoch + 1}, loss {epoch_loss:f}') 

# test
model.eval()
test_features = features[n_train:]
prediction = []
for i in range(len(test_features) // batch_size + 1):
  X = test_features[i * batch_size:(i + 1) * batch_size, :]
  X = torch.transpose(X, 1, 0)
  output = model(X)
  prediction += output.tolist()

# plot
plt.plot(timestamps, labels, label='original')
plt.plot(timestamps[n_train:], prediction, label='prediction')
plt.legend()
plt.xticks(rotation=30, fontsize=7)
plt.show()
