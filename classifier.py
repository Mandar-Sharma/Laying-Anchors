import argparse
 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-X", "--Xtrainpath", help="Tokenizer Folder")
parser.add_argument("-y", "--ytrainpath", help="Base Model Folder")
parser.add_argument("-Xtest", "--Xtestpath", help="Model Folder")
parser.add_argument("-ytest", "--ytestpath", help="Model Type")
args = parser.parse_args()
config = vars(args)


import torch
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


class LSTM(torch.nn.Module) :
    def __init__(self, embedding_dim, hidden_dim) :
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = 4, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, (ht, ct) = self.lstm(x)
        return self.act(self.linear(ht[-1]))


class NumeralDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)


def loss_batch(model, xb, yb, opt=None):
    preds = model(xb)
    loss = F.cross_entropy(preds, yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def evaluate(model, valid_dl):
    with torch.no_grad():
        results = [loss_batch(model, xb, yb) for xb, yb in valid_dl]
        losses, nums = zip(*results)
        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums)) / total
    return avg_loss, total



def fit(epochs, lr, model, train_dl, valid_dl):
    
    min_loss = float('inf')
    
    losses = []
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in tqdm(range(epochs)):
        for xb, yb in train_dl:
            loss,_ = loss_batch(model, xb, yb, opt)
        result = evaluate(model, valid_dl)
        val_loss, total = result
        
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), './C1.pt')
        
        losses.append(val_loss)
        
        if epoch % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, val_loss))
        
    return losses

with open(config['Xtrainpath'], 'rb') as fp:
    X_train = pickle.load(fp)
with open(config['ytrainpath'], 'rb') as fp:
    y_train = pickle.load(fp)
with open(config['Xtestpath'], 'rb') as fp:
    X_test = pickle.load(fp)
with open(config['ytestpath'], 'rb') as fp:
    y_test = pickle.load(fp)


X_train = np.array([np.array(x) for x in X_train])
X_test = np.array([np.array(x) for x in X_test])
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)

y_train = np.array([y.index(1) for y in y_train])
y_test = np.array([y.index(1) for y in y_test])

train_dataset = NumeralDataset(X_train, y_train)
test_dataset = NumeralDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)

device = torch.device("cuda")
train_dl = DeviceDataLoader(train_loader, device)
valid_dl = DeviceDataLoader(test_loader, device)

model = LSTM(264, 528)
model.cuda()
model.to(device)

losses = fit(150, 1e-4, model, train_dl, valid_dl)

def accuracy(model, valid_dl):
    acc = []
    for xb, yb in valid_dl:
        preds = model(xb)
        preds = np.argmax(preds.detach().cpu().numpy() > 0.5, axis=1)
        acc.append(accuracy_score(yb.detach().cpu().numpy(), preds))
    return np.mean(acc)

model_eval = LSTM(264, 528)
model_eval.load_state_dict(torch.load('./C1.pt'))
model_eval.cuda()
model_eval.to(device)       

print(accuracy(model, valid_dl))