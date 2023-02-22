import pandas as pd
data = pd.read_csv('Tesla_stock_Price.csv')
x = data["Date"]
y = data["Price"]

x = x.astype('float32')
y = y.astype('float32')

print(x)
print(y)

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

model = Sequential()  #define model
model.add(Dense(10, input_dim=1, activation="sigmoid"))  #add layers
model.add(Dense(10, activation="sigmoid"))
model.add(Dense(1, activation="tanh"))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"]) #compile model

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

with tf.device('/gpu:0'):
    model.fit(x,y, epochs=500, batch_size=10)  #training

_, accuracy = model.evaluate(x,y)    #testing
print("Model accuracy: %.2f"% (accuracy*100))

predictions = model.predict(x)     #make predictions

#round the prediction
rounded = [round(x[0]) for x in predictions]

#compare the prediction with the actual value
correct = 0
for (actual, prediction) in zip(y, rounded):
    if prediction != actual:
        print("Actual: %s, Predicted: %s (incorrect)" % (actual, prediction))
    else:
        print("Actual: %s, Predicted: %s (correct)" % (actual, prediction))
        correct += 1

print("Total incorrect: %s" % correct)
print("Accuracy: %.2f%%" % (correct / len(y) * 100))

random_prime = 3258

print("Random date: %s" % random_prime)
manual_prediction_prime = model.predict([random_prime])
print("Random prediction: %s" % manual_prediction_prime)
try:
    model.save("models/tesla_predict.v1.h5")
except:
    model.save("models/")