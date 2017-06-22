print("Doing the machine learning...")

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("testMakeathon.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=30, batch_size=10)

# evaluate the model
# using the same data set as the input (training) set for simplicity but should use other set ideally
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
testdataset = numpy.loadtxt("testMakeathonTest.csv", delimiter=",")
# split into input (X) and output (Y) variables
testX = testdataset[:,0:8]
# testY = testdataset[:,8]
predictions = model.predict(testX)
print(predictions)

# round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)