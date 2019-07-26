import codecs
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import requests
#%%
df = pd.read_csv("final.csv")
#%%
df = df.set_index('time')
#%%
x = df.drop('t+1', axis = 1)
y = df['t+1']
#%%
y=np.reshape(y.values, (-1,1))

#%%
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(x))
xscale=scaler_x.transform(x)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)
#%%

sData = xscale[0:1]
jack = x.head(1)

my_dict = dict()
for i, k in zip(jack,sData[0][:]):
    my_dict[i] = k

resp = requests.post('http://127.0.0.1:5000/predict', data = json.dumps(my_dict))
response = resp.json()
print(response)
ynew = scaler_y.inverse_transform(yscale[0:1])
print(" Real=%s" % (ynew))