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

As I don't have much RAM and only have one GPU, I adapted the project to my setup. I only load one of the seven files and my dataset is only composed of 15000 samples. Each training input is composed of the entity we want to predict with a "<norm>" token on each side to indicate the focus on this part, plus a context window of three entities each before and after the block to normalize.
