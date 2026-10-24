import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic

X_train = np .array ([[1.0] ,[2.0]], dtype = np.float32)
Y_train = np .array ([[300.0] ,[500.0]], dtype = np.float32)
fig ,ax = plt.subplots (1,1)
ax.scatter(X_train, Y_train,marker='o', color='blue',label = "Data Points")
ax.set_xlabel("X")
ax.set_ylabel("Y") 
ax.legend()
plt.show()

    
linear_layer = tf.keras.layers.Dense(units=1, activation='linear')
linear_layer.get_weights()

a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)

w,b= linear_layer.get_weights()
print(f"w = {w}, b ={b}")

set_w =np.array([[200]])
set_b =np.array([100])
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())
a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)
alin = np.dot(set_w, X_train[0].reshape(1,1)) + set_b
print(alin)
prediction_tf = linear_layer(X_train)
prediction_np = np.dot(X_train, set_w) + set_b
plt_linear(X_train, Y_train, prediction_tf, prediction_np)




