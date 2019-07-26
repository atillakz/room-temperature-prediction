# Load libraries
import json

import flask
from  flask import request, jsonify
from keras.models import load_model
import numpy as np
import tensorflow as tf
import pandas as pd
#%%
from sklearn.preprocessing import MinMaxScaler

def request_parameters():

   data = request.get_json(force=True)

   temp = data.pop('t')

   cond1_off = data.pop('cond1_off')

   cond1_on16h = data.pop('cond1_on16h')

   cond1_on16l = data.pop('cond1_on16l')

   cond1_on23h = data.pop('cond1_on23h')

   cond1_on23l = data.pop('cond1_on23l')

   cond1_on30l = data.pop('cond1_on30l')

   cond1_on30m = data.pop('cond1_on30m')

   cond2_off = data.pop('cond2_off')

   cond2_on16h = data.pop('cond2_on16h')

   cond2_on16l = data.pop('cond2_on16l')

   cond2_on16m = data.pop('cond2_on16m')

   cond2_on23h = data.pop('cond2_on23h')

   cond2_on23l = data.pop('cond2_on23l')

   cond2_on23m = data.pop('cond2_on23m')

   cond2_on30h = data.pop('cond2_on30h')

   cond2_on30l = data.pop('cond2_on30l')

   cond2_on30m = data.pop('cond2_on30m')

   x_data = np.array([[temp,
                       cond1_off,
                       cond1_on16h,
                       cond1_on16l,
                       cond1_on23h,
                       cond1_on23l,
                       cond1_on30l,
                       cond1_on30m,
                       cond2_off,
                       cond2_on16h,
                       cond2_on16l,
                       cond2_on16m,
                       cond2_on23h,
                       cond2_on23l,
                       cond2_on23m,
                       cond2_on30h,
                       cond2_on30l,
                       cond2_on30m]])
   return x_data


df = pd.read_csv("final.csv")
#%%
df = df.set_index('time')
#%%
x = df.drop('t+1', axis = 1)
y = df['t+1']
#%%
y=np.reshape(y.values, (-1,1))

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(x))
xscale=scaler_x.transform(x)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)
# instantiate flask
app = flask.Flask(__name__)

model_1 = load_model('./models/xor_model_1')
model_2 = load_model('./models/xor_model_2')
model_3 = load_model('./models/xor_model_3')
model_4 = load_model('./models/xor_model_4')
model_5 = load_model('./models/xor_model_5')
model_6 = load_model('./models/xor_model_6')
model_7 = load_model('./models/xor_model_7')
model_8 = load_model('./models/xor_model_8')
model_9 = load_model('./models/xor_model_9')
model_10 = load_model('./models/xor_model_10')

graph = tf.get_default_graph()

# define a predict function as an endpoint
@app.route("/predict/5", methods=["GET", "POST"])
def predict():

    u_data = request_parameters()
    with graph.as_default():
        result = model_5.predict(u_data)[0].tolist()
        f_re = np.array([result])
        ynew = scaler_y.inverse_transform(np.array(f_re))
        np_array_to_list = ynew.tolist()
        my_data = json.dumps(np_array_to_list)
        data = {'result': my_data}
        print(data)
    return jsonify(data)

@app.route("/predict/1", methods=["GET", "POST"])
def predict1():

    u_data = request_parameters()
    with graph.as_default():
        result = model_1.predict(u_data)[0].tolist()
        f_re = np.array([result])
        ynew = scaler_y.inverse_transform(np.array(f_re))
        np_array_to_list = ynew.tolist()
        my_data = json.dumps(np_array_to_list)
        data = {'result': my_data}
        print(data)
    return jsonify(data)
@app.route("/predict/2", methods=["GET", "POST"])
def predict2():

    u_data = request_parameters()
    with graph.as_default():
        result = model_2.predict(u_data)[0].tolist()
        f_re = np.array([result])
        ynew = scaler_y.inverse_transform(np.array(f_re))
        np_array_to_list = ynew.tolist()
        my_data = json.dumps(np_array_to_list)
        data = {'result': my_data}
        print(data)
    return jsonify(data)
@app.route("/predict/3", methods=["GET", "POST"])
def predict3():

    u_data = request_parameters()
    with graph.as_default():
        result = model_3.predict(u_data)[0].tolist()
        f_re = np.array([result])
        ynew = scaler_y.inverse_transform(np.array(f_re))
        np_array_to_list = ynew.tolist()
        my_data = json.dumps(np_array_to_list)
        data = {'result': my_data}
        print(data)
    return jsonify(data)

@app.route("/predict/4", methods=["GET", "POST"])
def predict4():

    u_data = request_parameters()
    with graph.as_default():
        result = model_4.predict(u_data)[0].tolist()
        f_re = np.array([result])
        ynew = scaler_y.inverse_transform(np.array(f_re))
        np_array_to_list = ynew.tolist()
        my_data = json.dumps(np_array_to_list)
        data = {'result': my_data}
        print(data)
    return jsonify(data)
@app.route("/predict/6", methods=["GET", "POST"])
def predict6():

    u_data = request_parameters()
    with graph.as_default():
        result = model_6.predict(u_data)[0].tolist()
        f_re = np.array([result])
        ynew = scaler_y.inverse_transform(np.array(f_re))
        np_array_to_list = ynew.tolist()
        my_data = json.dumps(np_array_to_list)
        data = {'result': my_data}
        print(data)
    return jsonify(data)

@app.route("/predict/7", methods=["GET", "POST"])
def predict7():

    u_data = request_parameters()
    with graph.as_default():
        result = model_7.predict(u_data)[0].tolist()
        f_re = np.array([result])
        ynew = scaler_y.inverse_transform(np.array(f_re))
        np_array_to_list = ynew.tolist()
        my_data = json.dumps(np_array_to_list)
        data = {'result': my_data}
        print(data)
    return jsonify(data)


@app.route("/predict/8", methods=["GET", "POST"])
def predict8():

    u_data = request_parameters()
    with graph.as_default():
        result = model_8.predict(u_data)[0].tolist()
        f_re = np.array([result])
        ynew = scaler_y.inverse_transform(np.array(f_re))
        np_array_to_list = ynew.tolist()
        my_data = json.dumps(np_array_to_list)
        data = {'result': my_data}
        print(data)
    return jsonify(data)


@app.route("/predict/9", methods=["GET", "POST"])
def predict9():

    u_data = request_parameters()
    with graph.as_default():
        result = model_9.predict(u_data)[0].tolist()
        f_re = np.array([result])
        ynew = scaler_y.inverse_transform(np.array(f_re))
        np_array_to_list = ynew.tolist()
        my_data = json.dumps(np_array_to_list)
        data = {'result': my_data}
        print(data)
    return jsonify(data)


@app.route("/predict/10", methods=["GET", "POST"])
def predict10():

    u_data = request_parameters()
    with graph.as_default():
        result = model_10.predict(u_data)[0].tolist()
        f_re = np.array([result])
        ynew = scaler_y.inverse_transform(np.array(f_re))
        np_array_to_list = ynew.tolist()
        my_data = json.dumps(np_array_to_list)
        data = {'result': my_data}
        print(data)
    return jsonify(data)

# start the flask app, allow remote connections
app.run(host='0.0.0.0')