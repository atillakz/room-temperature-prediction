import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json


#%%
df = pd.read_csv("./data/final_1.csv")
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
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

model = Sequential()
model.add(Dense(50, input_dim=18, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


# Use a custom metricfunction

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x, y, epochs=150, batch_size=50,  verbose=1, validation_split=0.01)

# save model and weights to file
model.save('xor_model')





ynew= model.predict(X_test[0:1])
print(ynew)
ynew = scaler_y.inverse_transform(ynew)
print(" Predicted=%s" % (ynew))
my_test = scaler_y.inverse_transform(y_test)


