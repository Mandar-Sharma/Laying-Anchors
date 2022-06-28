import argparse
 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-X", "--Xtrainpath", help="Tokenizer Folder")
parser.add_argument("-y", "--ytrainpath", help="Base Model Folder")
parser.add_argument("-Xtest", "--Xtestpath", help="Model Folder")
parser.add_argument("-ytest", "--ytestpath", help="Model Type")
args = parser.parse_args()
config = vars(args)

import pickle
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer

model = XGBRegressor(n_estimators = 1000, max_depth = 5, learning_rate = 0.01, gamma=0, tree_method='gpu_hist', gpu_id=0)

def log_squash(nums):
    return_list = []
    for num in nums:
        if num > 1:
            return_list.append(np.log(num) + 1)
        elif num < -1:
            return_list.append(-np.log(-num) -1)
        else:
            return_list.append(num)
    return return_list

with open(config['Xtrainpath'], 'rb') as fp:
    X_train = pickle.load(fp)
with open(config['ytrainpath'], 'rb') as fp:
    y_train = pickle.load(fp)
with open(config['Xtestpath'], 'rb') as fp:
    X_test = pickle.load(fp)
with open(config['ytestpath'], 'rb') as fp:
    y_test = pickle.load(fp)

log_scaler = FunctionTransformer(log_squash)

X_train = np.array([x.detach().numpy() for x in X_train])
X_test = np.array([x.detach().numpy() for x in X_test])

y_train = log_scaler.fit_transform(y_train)
y_test = log_scaler.transform(y_test)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(mean_squared_error(y_test, y_pred, squared=False))