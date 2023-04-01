# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 20:01:15 2022

@author: guill
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import gc
import random
import pickle
import datetime

path="C:/Users/guill/Desktop/Devoirs/Devoir ESILV A5/Computational Intelligence Methods/dataset/"


def dataset():
    """
        Load the dataset in memory
        input: no input
        output: X: list of the inputs of the dataset for training
                Y: list of the outputs of the dataset for training
                countX: length of the longest input
                county: length of the longest output
    """
    reader = open(path+"output_1.csv", "r", encoding="utf8")
    reader.readline()
    x=[]
    y=[]
    X=[]
    Y=[]
    countX=0
    county=0
    for line in reader:
        if('eos' in line):
            pass
        else:
            if(line.count(',')==3):
                x.append(',')
            else:
                x.append(line.replace('"','').strip().split(',')[1])
                
            test=False
            if(line.replace('"','').strip().split(',')[0]=='PLAIN'):
                if(random.randint(1,16)==1):
                    test=True
            elif(line.replace('"','').strip().split(',')[0]=='PUNCT'):
                if(random.randint(1,6)==1):
                    test=True
            elif(line.replace('"','').strip().split(',')[0]=='ELECTRONIC'):
                pass
            else:
                test=True
            if(test):
                if(line.count(',')==3):
                    y.append('sil')
                else:
                    y.append(line.replace('"','').strip().split(',')[2])
            else:
                y.append(None)
        if(len(x)==15000):
            break
    reader.close()
    
    count=0
    for i in range(len(x)):
        if(y[i]!=None):
            X.append([])
            for j in range(1,4):
                if(i-j<0):
                    X[count].insert(0,'NULL')
                elif(x[i-j]=='.'):
                    X[count].insert(0,'NULL')
                elif(len(X[count])>0 and X[count][0]=='NULL'):
                    X[count].insert(0,'NULL')
                else:
                    memory=[]
                    for word in x[i-j].split(' '):
                        memory+=list(word)+[' ']
                    X[count]=memory+X[count]
            X[count].append('<norm>')
            for word in x[i].split(' '):
                X[count]+=list(word)+[' ']
            X[count]=X[count][:len(X[count])-1]
            X[count]+=['<norm>', ' ']
            for j in range(1,4):
                if(i+j>=15000):
                    X[count].append('NULL')
                elif(x[i+j]=='.'):
                    X[count].append('NULL')
                elif(len(X[count])>0 and X[count][len(X[count])-1]=='NULL'):
                    X[count].append('NULL')
                else:
                    for word in x[i+j].split(' '):
                        X[count]+=list(word)+[' ']
            X[count]=X[count][:len(X[count])-1]
            Y.append(y[i].split(' '))
            countX=max(countX, len(X[count]))
            county=max(county, len(Y[count]))
            count+=1
        
    return X, Y, countX, county

def buildVocabulary(data):
    """
        Make a tokenizer dictionary with data
        input: data: list of list of characters (input) or words (output)
        output: vocab: dictionary that links input characters or output words to a token
    """
    count=1
    vocab={}
    for i in range(len(data)):
        for word in set(data[i]):
            if(word not in list(vocab.keys()) and word!='NULL'):
                vocab[word]=count
                count+=1
    return vocab

def buildReverseVocabulary(vocab):
    """
        Reverse a tokenizer dictionary
        input: vocab: a token dictionary
        output: rev_vocab: dictionary that links tokens to input characters or output words
    """
    rev_vocab = {}
    for key in vocab.keys():
        rev_vocab[vocab[key]]=key
    return rev_vocab

def wordsToIntArrayInput(data, vocab, length):
    """
        Tokenize inputs with input_vocab
        input: data: list of list of characters (each sublist is an input)
               vocab: a token dictionnary (input_vocab)
               length: length of the longest input
        output: data: list of list of tokens (inputs tokenized)
    """
    for i in range(len(data)):
        for j in range(len(data[i])):
            if(data[i][j]=='NULL'):
                data[i][j]=0
            else:
                data[i][j]=vocab[data[i][j]]
        data[i]+=list(np.zeros(length-len(data[i])))
    return data

def wordsToIntArrayOutput(data, vocab, length):
    """
        Tokenize inputs with input_vocab
        input: data: list of list of words (each sublist is an output)
               vocab: a token dictionnary (output_vocab)
               length: length of the longest output
        output: data: list of list of tokens (outputs tokenized)
    """
    wordSeq = np.zeros((len(data), length, len(vocab)))
    for i, seq in enumerate(data):
        for j in range(length):
            if(j<len(seq)):
                wordSeq[i, j, vocab[seq[j]]] = 1
            else:
                wordSeq[i, j, vocab['NULL']] = 1
    return wordSeq
        
def model(input_vocab, output_vocab, inlength, outlength):
    """
        Build the neural network
        input: input_vocab: dictionary that links input characters to a token
               output_vocab: dictionary that links output words to a token
               inlength: length of the longest input
               outlength: length of the longest output
        output: model: a tf model
    """
    Neurons=256
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(len(input_vocab)+1, Neurons, input_length=inlength, mask_zero=True))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(Neurons, return_sequences = True), merge_mode = 'concat'))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(Neurons, return_sequences = True), merge_mode = 'concat'))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(Neurons), merge_mode = 'concat'))
    model.add(tf.keras.layers.RepeatVector(int(outlength)))
    
    model.add(tf.keras.layers.LSTM(Neurons, return_sequences=True))
    model.add(tf.keras.layers.LSTM(Neurons, return_sequences=True))
    model.add(tf.keras.layers.LSTM(Neurons, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(output_vocab))))
    model.add(tf.keras.layers.Activation('softmax'))

    #model.summary()

    return model

def compilate_model(model, Xtrain, Ytrain, Xtest, Ytest):
    """
        Set the parameters (callbacks, learning rate...) and run the neural network 
        input: model: a tf model
               Xtrain: list of 80% of the inputs of the dataset for training
               Ytrain: list of 80% of the outputs of the dataset for training
               Xtest: list of 20% of the inputs of the dataset to test
               Ytest: list of 20% of the outputs of the dataset for test 
        output: history: Samples of Accuracy and loss to evaluate the training
                saved_model: trained model
    """
    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='auto', verbose=0, patience=8)
    mc = tf.keras.callbacks.ModelCheckpoint(path+'best_model.h5', monitor='val_accuracy', mode='auto', verbose=0, save_best_only=True)
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)
    profile_logs = path + 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir = profile_logs, histogram_freq = 1)
    model.compile(optimizer=optimizer, loss ='categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(Xtrain,Ytrain, validation_data=(Xtest,Ytest), epochs=300, batch_size=16, verbose=0, callbacks=[es, mc, tensorboard])
    saved_model = tf.keras.models.load_model(path+'best_model.h5')
    return history, saved_model

def smooth_curve(points, factor=0.5):
    """
        Smooth the plot of the neural network results (accuracy, loss) to focus on important fluctuations
        input: points: Accuracy or Loss points from neural network history
        output: smoothed_points: Accuracy or Loss points transformed
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def show_results(history):
    """
        Show the graphs of Accuracy and Loss with matplotlib
        input: history: Samples of Accuracy and loss to evaluate the training
        output: no output
    """
    plt.plot(smooth_curve(history.history['accuracy']), label='train')
    plt.plot(smooth_curve(history.history['val_accuracy']), label='test')
    plt.title('accuracy:', pad=-80)
    plt.show()
    plt.plot(smooth_curve(history.history['loss']), label='train')
    plt.plot(smooth_curve(history.history['val_loss']), label='test')
    plt.title('loss:', pad=-80)
    plt.show()

def predict(sentence, inlength, input_vocab, rev_vocab):
    """
        Predict a normalization for a sentence not in the dataset
        input: sentence: string in natural language to normalize
               inlength: length of the longest input
               input_vocab: dictionary that links input characters to a token
               rev_vocab: dictionary that links tokens to an output word (based on output_vocab)
        output: prediction: string in natural language normalized
    """
    saved_model = tf.keras.models.load_model(path+'best_model.h5')
    data = sentence.strip().split(' ')
    input_data = []
    count=0
    for i in range(len(data)):
        input_data.append([])
        for j in range(1,4):
            if(i-j<0):
                input_data[count].insert(0,'NULL')
            elif(data[i-j]=='.'):
                input_data[count].insert(0,'NULL')
            elif(len(input_data[count])>0 and input_data[count][0]=='NULL'):
                input_data[count].insert(0,'NULL')
            else:
                memory=[]
                for word in data[i-j].split(' '):
                    memory+=list(word)+[' ']
                input_data[count] = memory + input_data[count]
        input_data[count].append('<norm>')
        for word in data[i].split(' '):
            input_data[count]+=list(word)+[' ']
        input_data[count]=input_data[count][:len(input_data[count])-1]
        input_data[count]+=['<norm>', ' ']
        for j in range(1,4):
            if(i+j>=len(data)):
                input_data[count].append('NULL')
            elif(data[i+j]=='.'):
                input_data[count].append('NULL')
            elif(len(input_data[count])>0 and input_data[count][len(input_data[count])-1]=='NULL'):
                input_data[count].append('NULL')
            else:
                for word in data[i+j].split(' '):
                    input_data[count]+=list(word)+[' ']
        input_data[count] = input_data[count][:len(input_data[count])-1]
        count+=1
        
    input_data = wordsToIntArrayInput(input_data, input_vocab, inlength)
    prediction=""
    pred = saved_model.predict(input_data)
    for p in pred:
        for i in range(len(p)):
            if(rev_vocab[list(p[i]).index(max(p[i]))] != 'NULL'):
                prediction += rev_vocab[list(p[i]).index(max(p[i]))] + ' '
                
    return prediction
    

    

if __name__=='__main__':
    start=time.time()
    
    """with tf.device('/device:CPU:0'):        #train the model
        X, y, inlength, outlength = dataset()
        input_vocab = buildVocabulary(X)
        output_vocab = buildVocabulary(y)
        output_vocab['NULL'] = 0
        X = wordsToIntArrayInput(X, input_vocab, inlength)
        y = wordsToIntArrayOutput(y, output_vocab, outlength)
    
        #X,y=shuffle(X,y)
        Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y, test_size=0.2)
        del X
        del y
        gc.collect()
        Xtrain=np.asarray(Xtrain)
        Ytrain=np.asarray(Ytrain)
        Xtest=np.asarray(Xtest)
        Ytest=np.asarray(Ytest)
        
        model = model(input_vocab, output_vocab, inlength, outlength)
    history, model = compilate_model(model, Xtrain, Ytrain, Xtest, Ytest)
    show_results(history)
    with open(path+"input_vocab.pickle", "wb") as f:
        pickle.dump(input_vocab, f)
    with open(path+"output_vocab.pickle", "wb") as f:
        pickle.dump(output_vocab, f)
    with open(path+"inlength.pickle", "wb") as f:
        pickle.dump(inlength, f)"""
        
    
    with open(path+"input_vocab.pickle", "rb") as f:
        input_vocab=pickle.load(f)
    with open(path+"output_vocab.pickle", "rb") as f:
        output_vocab=pickle.load(f)
    with open(path+"inlength.pickle", "rb") as f:
        inlength=pickle.load(f)
    rev_vocab = buildReverseVocabulary(output_vocab)
    prediction = predict("a baby giraffe is 8 ft tall and weights 120 lb .", inlength, input_vocab, rev_vocab)
    print(prediction)
    
    
    print(time.time()-start, "sec")
    
    gc.collect()