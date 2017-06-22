print("Doing the machine learning...")

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

LAYERS = [8,16,16,1]

# load pima indians dataset
dataset = numpy.loadtxt("MakeathonAccept.csv", delimiter=",")
# sex;age;address;famsize;Pstatus;Medu;Fedu;Mjob;Fjob;reason;guardian;
# traveltime;studytime;failures;schoolsup;famsup;paid;activities;nursery;higher;
# internet;romantic;famrel;freetime;goout;Dalc;Walc;health;absences;G1;G2;G3
# split into input (X) and output (Y) variables
X = dataset[:,0:LAYERS[0]]
Y = dataset[:,LAYERS[0]:LAYERS[0] + 1]

# create model
model = Sequential()
model.add(Dense(LAYERS[0], input_dim=LAYERS[0], activation='relu'))
model.add(Dense(LAYERS[1], activation='relu'))
model.add(Dense(LAYERS[2], activation='relu'))
model.add(Dense(LAYERS[3], activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=100, batch_size=10, shuffle=True, validation_split=.25)

# evaluate the model
# using the same data set as the input (training) set for simplicity but should use other set ideally
testdataset = numpy.loadtxt("MakeathonFinalTest.csv", delimiter=",")
testX = testdataset[:,0:LAYERS[0]]
testY = testdataset[:,LAYERS[0]]

# scores = model.evaluate(testX, testY)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Predict your chance of getting into CoLab!
name = input('Enter your name: ')
print('Hello', name)
gradYear = input('When did or will you graduate? ')
levelEdu = input('What is your level of education? (0 = undergrad, 1 = master, 2 = doctoral, 3 = no school) ')
skill = input('Are you a Business Designer(0), Designer(1), Technologist(2), or something else(3)?')
craft = input('What is your craft score? (inputted by HR)')
r1 = input('What is your Resume Score? (NLP)')
r2 = input('What is your Cover Letter Score? (NLP)')
totalR = r1+r2
totalAll = totalR + craft

# calculate predictions
# testY = testdataset[:,8]
predictions = model.predict(testX, batch_size=1, verbose=0)
print(predictions)
print(testX)
# round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)