# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 20:01:15 2022

@author: guill
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}

import time
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import gc
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import random
import pickle
import datetime

path="C:/Users/guill/Desktop/Devoirs/Devoir ESILV A5/Computational Intelligence Methods/dataset/"

def dataset():
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
                if(random.randint(1,10)==1):
                    test=True
            elif(line.replace('"','').strip().split(',')[0]=='PUNCT'):
                if(random.randint(1,8)==1):
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
        if(len(x)==10000):
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
                    X[count]=x[i-j].split(' ')+X[count]
            X[count].append('<norm>')
            X[count]+=x[i].split(' ')
            X[count].append('</norm>')
            for j in range(1,4):
                if(i+j>=10000):
                    X[count].append('NULL')
                elif(x[i+j]=='.'):
                    X[count].append('NULL')
                elif(len(X[count])>0 and X[count][len(X[count])-1]=='NULL'):
                    X[count].append('NULL')
                else:
                    X[count]+=x[i+j].split(' ')
            Y.append(y[i].split(' '))
            countX=max(countX, len(X[count]))
            county=max(county, len(Y[count]))
            count+=1
        
    return X, Y, countX, county

def datasetbis():
    reader = open(path+"output_1.csv", "r", encoding="utf8")
    reader.readline()
    X=[["NULL","NULL"],["NULL"],[]]
    y=[[],[],[]]
    count=0     #nb of sentences in the dataset
    precountX=[0,0,0,0,0]
    precounty=[0,0,0,0,0]
    countX=0
    county=0
    classes={}
    for line in reader:
        if('eos' in line):
            pass
        else:
            if(line.count(',')==3):
                memoryX=[',']
                memoryy=['sil']
            #elif('self' in line):
                #memoryX=line.replace('"','').strip().split(',')[1].split(' ')
                #memoryy=line.replace('"','').strip().split(',')[1].split(' ')
            else:
                memoryX=line.replace('"','').strip().split(',')[1].split(' ')
                memoryy=line.replace('"','').strip().split(',')[2].split(' ')
            for i in range(-2,3):
                if(count+i<0):
                    pass
                elif(i==0):
                    test=False
                    if(line.replace('"','').strip().split(',')[0]=='PLAIN'):
                        if(random.randint(1,5)==1):
                            test=True
                    elif(line.replace('"','').strip().split(',')[0]=='PUNCT'):
                        if(random.randint(1,5)==1):
                            test=True
                    elif(line.replace('"','').strip().split(',')[0]=='ELECTRONIC'):
                        pass
                    else:
                        test=True
                    if(test):
                        X[count+i]+=list("<norm> ")
                        precountX[2+i]+=len(list("<norm> "))
                        for word in memoryX:
                            X[count+i]+=list(word+' ')
                            precountX[2+i]+=len(list(word+' '))
                        X[count+i]+=list("</norm> ")
                        precountX[2+i]+=len(list("</norm> "))
                        for word in memoryy:
                            y[count+i]+=[word]+[' ']
                            precounty[2+i]+=2
                        y[count+i]=y[count+i][:len(y[count+i])-1]
                        precounty[2+i]-=1
                        if line.replace('"','').strip().split(',')[0] in classes.keys():
                            classes[line.replace('"','').strip().split(',')[0]]+=1
                        else:
                            classes[line.replace('"','').strip().split(',')[0]]=1
                else:
                    for word in memoryX:
                        X[count+i]+=list(word+' ')
                        precountX[2+i]+=len(list(word+' '))
            X[count-2]=X[count-2][:len(X[count-2])-1]
            precountX[0]-=1
            if(count-2>=0):
                if(precountX[0]>countX):
                    countX=precountX[0]
                if(precounty[0]>county):
                    county=precounty[0]
                precountX.pop(0)
                precounty.pop(0)
                precountX.insert(len(precountX),0)
                precounty.insert(len(precounty),0)
            count+=1
            X.append([])
            y.append([])
            if(count-3>=0):
                if('<' not in X[count-3]):
                    X.pop(count-3)
                    y.pop(count-3)
                    count-=1
            if(count==5002):
                break
    reader.close()
    return X[:len(X)-5],y[:len(y)-5],count-2,countX,county,classes

def datasetter():
    reader = open(path+"output_1.csv", "r", encoding="utf8")
    reader.readline()
    x=[]
    y=[]
    X=[]
    Y=[]
    countX=0
    county=0
    classes={}
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
    count=1
    vocab={}
    for words in set(data):
        for word in words.split(" "):
            vocab[word]=count
            count+=1
    return vocab

def buildVocabularybis(data):
    count=1
    vocab={}
    for i in range(len(data)):
        for word in set(data[i]):
            if(word not in list(vocab.keys()) and word!='NULL'):
                vocab[word]=count
                count+=1
    return vocab

def buildReverseVocabulary(vocab):
    rev_vocab = {}
    for key in vocab.keys():
        rev_vocab[vocab[key]]=key
    return rev_vocab

def wordsToIntArray(data, vocab, length):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if(data[i][j]=='NULL'):
                data[i][j]=0
            else:
                data[i][j]=vocab[data[i][j]]
        data[i]+=list(np.zeros(length-len(data[i])))
    return data

def wordsToIntArraybis(data, vocab, length):
    wordSeq = np.zeros((len(data), length, len(vocab)))
    for i, seq in enumerate(data):
        for j in range(length):
            if(j<len(seq)):
                wordSeq[i, j, vocab[seq[j]]] = 1
            else:
                wordSeq[i, j, vocab['NULL']] = 1
    return wordSeq
    
        
def model(input_vocab, output_vocab, inlength, outlength):
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
    loss_history = LossHistory()
    lrate = tf.keras.callbacks.LearningRateScheduler(exp_decay)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='auto', verbose=0, patience=8)
    mc = tf.keras.callbacks.ModelCheckpoint(path+'best_model.h5', monitor='val_accuracy', mode='auto', verbose=0, save_best_only=True)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=1, decay_rate=0.9)
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)
    profile_logs = path + 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir = profile_logs, histogram_freq = 1)
    model.compile(optimizer=optimizer, loss ='categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(Xtrain,Ytrain, validation_data=(Xtest,Ytest), epochs=300, batch_size=16, verbose=0, callbacks=[es, mc, tensorboard])
    saved_model = tf.keras.models.load_model(path+'best_model.h5')
    return history, saved_model

def smooth_curve(points, factor=0.5):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def show_results(history):
    plt.plot(smooth_curve(history.history['accuracy']), label='train')
    plt.plot(smooth_curve(history.history['val_accuracy']), label='test')
    plt.title('accuracy:', pad=-80)
    plt.show()
    plt.plot(smooth_curve(history.history['loss']), label='train')
    plt.plot(smooth_curve(history.history['val_loss']), label='test')
    plt.title('loss:', pad=-80)
    plt.show()
    
def step_decay(epoch):
    """
        Decrease the learning rate following a step mathematical function
        input: epoch: State of training of the network
        output: lrate: parameter of the gradient descent
    """
    initial_lrate = 0.005
    drop = 0.0001
    epochs_drop = 6
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def exp_decay(epoch):
    """
        Decrease the learning rate following a exponential mathematical function
        input: epoch: State of training of the network
        output: lrate: parameter of the gradient descent
    """
    initial_lrate = 0.005
    k = 0.5
    lrate = initial_lrate * math.exp(-k*epoch)
    return lrate
    
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))
    

def predict(sentence, inlength, input_vocab, rev_vocab):
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
        
    input_data = wordsToIntArray(input_data, input_vocab, inlength)
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
        X, y, inlength, outlength = datasetter()
        input_vocab = buildVocabularybis(X)
        output_vocab = buildVocabularybis(y)
        output_vocab['NULL'] = 0
        X = wordsToIntArray(X, input_vocab, inlength)
        y = wordsToIntArraybis(y, output_vocab, outlength)
    
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
    prediction = predict("The prime minister , Donald Trump , lives at the 10 Downing Street .", inlength, input_vocab, rev_vocab)
    print(prediction)
    
    
    print(time.time()-start, "sec")
    
    gc.collect()