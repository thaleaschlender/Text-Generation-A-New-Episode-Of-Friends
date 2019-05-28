from keras.utils import np_utils
import numpy
import re
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Embedding
from keras.layers import LSTM
from keras.layers import RNN
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD
from keras.callbacks import LambdaCallback, ModelCheckpoint
import random
import sys
from keras import optimizers
import io
t = pd.read_pickle('AllScripts.pkl')
t = t.values.tolist()
text = []
seedInput = t[50]

for i in range(0,5):
    text.append(t[i])

def preprocess(text):
    text = [t[0].lower() for t in text]
    return text


def buildVocabulary(text):
    textchars = []
    for t in text:
        for c in t:
            #if c != ' ':
            textchars.append(c)
    temp = []
    for t in textchars:
        temp.append(t)
    print(type(temp), type(temp[0]), type(textchars),type(textchars[0]))

    temp.sort()

    vocab = dict((c, i) for i, c in enumerate(temp))
    count = 0
    for key in vocab.keys():
        vocab[key] = count
        count += 1

    vocab2 = dict((i, c) for i, c in enumerate(vocab))

    print(vocab)
    print(vocab2)
    return vocab, vocab2, textchars

def gettrainingdata(textchars,vocab):
    seedSize = 100
    dataX = []
    dataY = []
    trainingdata = []
    a = 0
    while (a + seedSize) < len(textchars):
        x = []
        for i in range(a, a + seedSize):
            #print(vocab.get(textchars[i]))
            x.append(vocab.get(textchars[i]))
        y = vocab.get(textchars[a + seedSize])
        temp = [x, y]
        trainingdata.append(temp)
        a = a + 1
    print("number of training pairs", len(trainingdata))
    for e in trainingdata:
        dataX.append(e[0])
        dataY.append(e[1])

    # reshape data y to one hot encoding
    dataY = np_utils.to_categorical(dataY)
    # reshape dataX for keras purposes: samples, timestep, features
    dX = numpy.reshape(dataX, (len(dataX),seedSize,1))
    # make values between 0-1
    dX = dX/float(len(vocab))
    return dX, dataY, dataX

def buildModel (X, y):
    model = Sequential()
    model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=250, activation='tanh'))
    #model.add(Dropout(0.15))
    #model.add(Dense(output_dim=80, activation='tanh'))
    model.add(Dense(y.shape[1], activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    print(model.summary())
    return model,sgd
def trainmodel(X,y, model):
    # define the checkpoint
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # fit the model
    model.fit(X, y, epochs=1000, batch_size=60, callbacks=callbacks_list)

def buildseed(text,vocab):
    text = [t.lower() for t in text]
    textchars = []
    for t in text:
        for c in t:
            # if c != ' ':
            textchars.append(c)
    seed = []
    print("LENGTH", len(textchars))
    f = 9420
    for j in range(f, f+100):
        seed.append(vocab.get(textchars[j]))
    print(seed)
    return seed

def test(model, dataX, vocabnumchar, rmsprop,seed):
    filename = "weights-improvement-788-1.3104.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
    # pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = seed#dataX[start]
    print("Seed:")

    for val in pattern:
        temp = vocabnumchar.get(val)
        print(temp)


    print("\"", ''.join(vocabnumchar.get(value) for value in pattern), "\"")
    print("enter neural net")
    # generate characters
    # generate characters
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(vocabnumchar))
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = vocabnumchar.get(index)#list(vocab.keys())[list(vocab.values()).index(temp)]
        #seq_in = [vocab[value] for value in pattern]
        sys.stdout.write(result)
        sys.stdout.write('')
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        #print(pattern)

    print("\nDone.")


print("HELLO WORLD")
text =  preprocess(text)
#vocabCharNum has chars as keys and numbers as values
#ocabNumChar has numbers as values and chars as keys
vocabCharNum,vocabNumChar, textchar = buildVocabulary(text)
datax, datay , X= gettrainingdata(textchar,vocabCharNum)
model,rmsprop = buildModel(datax, datay)
#trainmodel(datax,datay,model)

seed = buildseed(seedInput,vocabCharNum)
test(model, X, vocabNumChar,rmsprop,seed)

