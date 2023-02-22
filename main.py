import pandas as pd
data = pd.read_csv('primes.csv')
x = data["num"]
y = data["is_prime"]

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

model = Sequential()  #define model
model.add(Dense(12, input_dim=1, activation="relu"))  #add layers
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) #compile model

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

with tf.device('/gpu:0'):
    model.fit(x,y, epochs=150, batch_size=10000)  #training

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

from random import randint
random_num = randint(1, 1000000000)
random_prime = 1824601

print("Random number: %s" % random_num)
print("Random prime: %s" % random_prime)
manual_prediction = model.predict([random_num])
manual_prediction_prime = model.predict([random_prime])
print("Random number prediction: %s" % manual_prediction)
print("Random prime prediction: %s" % manual_prediction_prime)
try:
    model.save("models/prime_model.v1.h5")
except:
    model.save("models/")