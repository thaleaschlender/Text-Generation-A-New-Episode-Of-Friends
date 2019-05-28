import numpy as np
import re
import pandas as pd
from nltk import tokenize
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Embedding
from keras.layers import LSTM
from keras.layers import RNN
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint
from string import punctuation
import random
import sys
import io

class Wordlevel:

    def __init__ (self):
        pass

    """
    Receives a string as input, cleans the string and outputs the cleaned string
    """
    def clean(self, sentence):

        #Regular Expressions
        sentence = re.sub("\'", "'", sentence)
        sentence = re.sub('(?<=[0-9])\,(?=[0-9])', "", sentence)
        sentence = re.sub('(?<=[0-9])\ (?=[0-9])', "", sentence)
        sentence = re.sub('-', " ", sentence)

        return sentence

    """
    Receives a pandas data frame with one column that is string type and outputs a list of strings
    """
    def DfToList(self, df):

        allScripts = []
        text = df.values.tolist()
        for t in text:
            for c in t:
                allScripts.append(c)
        return allScripts

    """
    Creates and prepares a sequential model with certain attributes
    """
    def prepareModel(self):

        model = Sequential()
        model.add(LSTM(100, input_shape=(self.X.shape[1], self.X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim=250, activation='tanh'))
        model.add(Dense(output_dim=500, activation='tanh'))
        model.add(Dense(output_dim=1000, activation='tanh'))
        model.add(Dense(self.y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(model.summary())

        return model


    def prepareData(self, List):

        allText = ""
        for episode in List:
            allText = allText + " " + self.clean(episode)

        allTextTokenized = word_tokenize(allText)
        tokenizer = Tokenizer(filters="")

        tokenizer.fit_on_texts(allTextTokenized)

        # turn words into sequences
        s = tokenizer.texts_to_sequences(allTextTokenized)
        self.word_idx = tokenizer.word_index
        self.idx_word = tokenizer.index_word
        self.num_words = len(self.word_idx) + 1
        t = tokenizer.sequences_to_texts(s)

        w = [''.join(i) for i in t]
        wseq = []
        for a in w:
            if a != '':
                wseq.append(self.word_idx[a])

        trainingdata = []
        a = 0
        while (a + 60) < len(wseq):
            x = []
            for i in range(a, a + 60):
                x.append(wseq[i])
            y = wseq[a + 60]
            temp = [x, y]
            trainingdata.append(temp)
            a = a + 1

        print("number of training pairs", len(trainingdata))
        print('number of words :', self.num_words)
        self.dataX = []
        self.dataY = []
        for e in trainingdata:
            self.dataX.append(e[0])
            self.dataY.append(e[1])

        # reshape X to be [samples, time steps, features]
        self.X = np.reshape(self.dataX, (len(trainingdata), 60, 1))
        # normalize
        self.X = self.X / float(self.num_words)
        # one hot encode the output variable
        self.y = np_utils.to_categorical(self.dataY)

    def train(self):

        model = self.prepareModel()

        filepath = "weights-improvement-5episodes-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        # fit the model
        model.fit(self.X, self.y, epochs=500, batch_size=60, callbacks=callbacks_list)

    def topfive(self,prediction):
        bestFive = []
        temppred = prediction.copy()
        for i in range(0, 5):

            best = np.argmax(temppred)
            # result = idx_word[best]
            prob = temppred[0,best]
            temppred = np.delete(temppred, best,1)
            bestFive.append([best, prob])
        #print(bestFive)
        return bestFive

    def beamSearch(self,model, prediction, pattern, num_words):
        # find best five prediction indexs and probabilities
        bestfive = self.topfive(prediction)
        best = [1, 1, 0]
        # for each of the best five predictions expand one level further
        for e in bestfive:
            tempPattern = pattern.copy()
            tempPattern.append(e[0])
            tempPattern = tempPattern[1:len(tempPattern)]
            x = np.reshape(tempPattern, (1, len(tempPattern), 1))
            x = x / float(num_words)
            tempPrediction = model.predict(x, verbose=0)
            tempbestfive = self.topfive(tempPrediction)
            t = e[1] + tempbestfive[0][1]
            tempBest = [e[0], tempbestfive[0][0], t]
            for a in tempbestfive:
                prob = e[1] + a[1]
                if prob > tempBest[2]:
                    tempBest = [e[0], a[0], prob]
            #print(tempBest)
            if tempBest[2] > best[2]:
                best = tempBest

        return best[0]

    def test(self, filename, seed):

        model = self.prepareModel()
        model.load_weights(filename)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        # pick a random seed
        start = np.random.randint(0, len(self.dataX) - 1)
        #pattern = self.dataX[start]

        seed = seed.lower()
        seed = word_tokenize(seed)
        seed = seed[0:60]
        seedInd = [self.word_idx[token] for token in seed]

        print("Seed:")
        print("\"", ' '.join([self.idx_word[value] for value in seedInd]), "\"")

        for i in range(250):
            x = np.reshape(seedInd, (1, len(seedInd), 1))
            x = x / float(self.num_words)
            prediction = model.predict(x, verbose=0)
            index = self.beamSearch(model,prediction,seedInd, self.num_words)
            #index = np.argmax(prediction)
            result = self.idx_word[index]
            seq_in = [self.idx_word[value] for value in seedInd]
            sys.stdout.write(result)
            sys.stdout.write(' ')
            seedInd.append(index)
            seedInd = seedInd[1:len(seedInd)]

        print("\nDone.")

    def main(self):

        df = pd.read_pickle('AllScripts.pkl')
        allScripts = self.DfToList(df)

        allScripts = allScripts[0:25]  # Limit the number of episodes



        self.prepareData(allScripts)

        #self.train()

        weights5 = "weights-improvement-5episodes-673-0.4354.hdf5"
        weights50 = "weights-improvement-191-3.3068.hdf5"

        seed1 = "[Scene: Central Perk, Chandler, Monica, Janice are sitting on the couch, and Phoebe is sitting next to them in the couch.] Chandler: Well, there are no good shows. Janice: Well, let's go to a bad one and make out. (they start to kiss and lean back into Monica.) Monica: Perhaps, you would like me to turn like this, (turns sideways on the couch) so that you can bunny bump against my back."
        seed2 = "[Scene: Chandler and Joey's, Chandler and Janice are having dinner] Janice: So, how come you wanted to eat in tonight? Chandler: 'Cause, I wanted to uh, give you this. (hands her a gift) Janice: Ohhh, are you a dog! (opens it) Contact paper! I never really know what to say when someone you're sleeping with gives you contact paper."
        seed3 = "[Scene: Monica and Rachel's, Rachel is getting ready for her first day.] Rachel: (coming in from her bedroom, wearing only a towel) Okay. Hey. Umm. Does everybody hate these shoes? Chandler: Oh yeah, but don't worry. I don't think anybody's gonna focus on that as long as your wearing that towel dress. Rachel: (to Ross) Tell him. Ross: (to Chandler) It's her first day at this new job. Your not supposed to start with her!"

        self.test(weights50, seed1)


Wordlevel().main()

