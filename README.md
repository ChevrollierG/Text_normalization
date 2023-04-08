# Text_normalization

### The project

This project consists on an implementation of an encoder-decoder LSTM neural network to perform text-to-speech normalization (convert a written sentence to a spoken form) with deep learning. It is based on the Kaggle competition **Google Text Normalization Challenge** that provides a research paper and a dataset in which we can find english and russian words (to work on both language), I only worked on the english language to make things a bit easier.

An example of Text-To-Speech Normalization  
<p align="center">
  <img src="https://user-images.githubusercontent.com/91634314/229577972-fa2e18cb-4d3b-463e-8fc9-ce62a2ca67e0.png?raw=true" alt="Sublime's custom image"/>
  <br>
  <i>An example of Text-To-Speech Normalization</i>
</p>

### Transforming the dataset

The dataset can be downloaded from Kaggle and comes in the form of 2GB splitted in 7 csv files. On each lines of these files, we can find one training sample that is composed of: it's class (type of entity it represents in the sentence), the input entity (the textual form) and the output one (the speech form).  
<p align="center">
  <img src="https://user-images.githubusercontent.com/91634314/229623367-ed7c5b6e-4212-4349-9448-0b4e233c020a.png?raw=true" alt="Sublime's custom image"/>
  <br>
  <i>3 samples from a file</i>
</p>

As I don't have much RAM and only have one GPU, I adapted the project to my setup. I only load one of the seven files and my dataset is only composed of 15000 samples. Each training input is composed of the entity we want to predict with a ```<norm>``` token on each side to indicate the focus on this part, plus a context window of three entities each before and after the block to normalize. We also use the token ```NULL``` to pad the input and the output that doesn't match the respective maximum length.

<p align="center">
  <img src="https://user-images.githubusercontent.com/91634314/229868740-d9c5df01-66ee-4e8e-a50c-02e744b286a4.png?raw=true" alt="Sublime's custom image"/>
  <br>
  <i>Input of the neural network</i>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/91634314/229876499-d1831560-0d36-40ea-b347-1e43530826ff.png?raw=true" alt="Sublime's custom image"/>
  <br>
  <i>Output of the neural network</i>
</p>

After setting up the input and the output, we want to tokenize (replace an entity in natural language by a number as the network only understands mathematics) our input character by character (except for ```NULL``` and ```<norm>``` which are not decomposed) and output by word.

### Neural network training

First we load the new dataset we made, shuffle it to have some diversity and split it between training (80%) and test (20%). For the validation we can write any sentence to test, the only problem we will encounter is if we enter a character not present in the input vocabulary or a word not in the output vocabulary.

Then we build the auto-encoder model that takes in input the input vocabulary, the output vocabulary, the maximum length of input and the maximum length of output. The encoder part is composed of three Bidirectional LSTM layers to take into account the preceding and following context, the decoder is three simple LSTM layers. Every LSTM layers are composed of 256 neurons.

To train the model, we use RMSprop optimizer and categorical_crossentropy loss function as we do classification and for the others (batch size, learning rate) I tested some until it was ok. Two callbacks are here to save the model everytime it gets better (following the validation accuracy) and stop the training if the model doesn't get better after a number of epochs which we defined as 8, another callback uses tensorboard to follow the metrics of the network (accuracy, loss) during the training.

Accuracy plot for the traning (blue is training and orange is validation)

<p align="center">
  <img src="https://user-images.githubusercontent.com/91634314/230719281-150abaa8-82c5-43c1-a92e-de67a6d69d74.PNG?raw=true" alt="Sublime's custom image"/>
  <br>
  <i>Ratio of accuracy following the number of epochs</i>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/91634314/230719333-ba00bc90-7969-48e9-8514-0e77d7a1850c.PNG?raw=true" alt="Sublime's custom image"/>
  <br>
  <i>Ratio of loss following the number of epochs</i>
</p>

