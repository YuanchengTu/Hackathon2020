from flask import Flask, render_template, request
import numpy as np
import pandas as pd
app = Flask(__name__)

import torch
import torchtext
from torchtext import datasets
import torch.optim as optim
import torch.nn as nn

import spacy 
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
vocabfile = './model/vocab.pt'
modelfile = './model/model_acc_0.890_epochs_05.pt'

class RNN(nn.Module): 

    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 output_dim, n_layers, bidirectional, dropout):
        
        super().__init__()   

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #multi-layer bidirectional RNN
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, 
                           bidirectional = bidirectional, dropout=dropout)        
        self.fc = nn.Linear(hidden_dim*2, output_dim)        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):        
        embedded = self.dropout(self.embedding(text))       
        output, hidden = self.rnn(embedded)       
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
       
        return self.fc(hidden.squeeze(0))

def prediction(sentence, vocabfile, modelfile):

    vocab = torch.load(vocabfile)
    model = torch.load(modelfile)
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    print(f'| prediction : {prediction.item():.4f} | comment: {sentence} | ')

    return prediction.item()

########################## Web server routing ##############################
@app.route('/')
def main():
	return render_template('allyship_detection.html')

#Input:  string 
#Output: toxicity score
@app.route('/predictToxicity', methods = ['GET'])
def getPredictedToxicity():
	sentence = request.args.get('sentence')
	score = prediction(sentence, vocabfile, modelfile)
	return f'{(1.0-score):.3f}'

if __name__ == '__main__':
	app.run()
    
