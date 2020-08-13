
from humanfriendly import format_timespan
import time
begin_time = time.time()

import argparse
from argparse import Namespace
import torch
import torchtext
from torchtext import datasets
import torch.optim as optim
import torch.nn as nn
import os
import re
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from azureml.core import Run
run = Run.get_context()
import spacy 
#os.system("python -m spacy download en")
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

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

def parse_script_args(parser):

    parser.add_argument("--data_dir",
                    default="./data/cleaned/",
                    type=str,
                    help="The input corpus dir.")
    parser.add_argument("--train_data",
                    default="demoTrain.csv",
                    type=str,
                    help="training corpus.")      
    parser.add_argument("--test_data",
                    default="demoTest.csv",
                    type=str,
                    help="testing corpus.") 
    parser.add_argument("--embedding_dir",
                        default=".vector_cache/",
                        type=str,
                        #required=True,
                        help="pre-trained embedding directory.")
    parser.add_argument("--embeddingFileName",
                        default="glove.6B.100d",
                        type=str,
                        #required=True,
                        help="pre-trained embedding file name.")
    parser.add_argument("--save_dir",
                        default="./model/",
                        type=str,
                        #required=True,
                        help="The output directory where the model file will be written.")
    parser.add_argument("--output_dir",
                        default="./output/",
                        type=str,
                        #required=True,
                        help="The output directory where the image file will be written.")

    # Add hypter parameters
    parser.add_argument("--num_epochs",
                        default='5',
                        type=int,
                        help="number of epochs")
    parser.add_argument("--embedding_dim",
                        default=100,
                        type=int,
                        help="embedding dimention.")
    parser.add_argument("--hidden_dim",
                        default=20,
                        type=int,
                        help="model hidden dimention.")
    parser.add_argument("--output_dim",
                        default=1,
                        type=int,
                        help="model output dimention.")
    parser.add_argument("--batch_size",
                        default=64,
                        type=int,
                        help="training batch size")
    parser.add_argument("--n_layers",
                        default='2',
                        type=int,
                        help="model layers")
    parser.add_argument("--bi_direction",
                        action='store_true',
                        dest = 'birnn',
                        help="bidirectional model")                        
    parser.add_argument("--dropout",
                        default=0.5,
                        type=float,
                        help="normalization drop out")

    return parser.parse_args()

def tokenizer(s): 
    return [w.text.lower() for w in nlp(s)]

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:       
        optimizer.zero_grad()       
        predictions = model(batch.comment).squeeze(1)        
        loss = criterion(predictions, batch.label)        
        rounded_preds = torch.round(torch.sigmoid(predictions))
        correct = (rounded_preds == batch.label).float()         
        acc = correct.sum() / len(correct)        
        loss.backward()        
        optimizer.step()        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

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

def main():
    parser = argparse.ArgumentParser()
    args =parse_script_args(parser) 

    #load dataset
    TEXT = torchtext.data.Field(tokenize = tokenizer)
    LABEL = torchtext.data.LabelField(dtype = torch.float)
    datafields = [('label', LABEL), ('comment', TEXT)]  
    trn, tst = torchtext.data.TabularDataset.splits(path = args.data_dir, 
                                                train = args.train_data,
                                                test = args.test_data,    
                                                format = 'csv',
                                                skip_header = True,
                                                fields = datafields)

    print(f'Number of training examples: {len(trn)}')
    print(f'Number of testing examples: {len(tst)}')
    # run.log("Number of training examples", len(trn))
    # run.log("Number of testing examples", len(tst))

    #embeddingfile = os.path.join(args.embedding_dir, args.embeddingFileName)
    TEXT.build_vocab(trn, max_size=25000,
                 vectors=args.embeddingFileName,
                 unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(trn)

    #print(TEXT.vocab.freqs.most_common(50))
    #print(TEXT.vocab.itos[:50])
    #print(LABEL.vocab.stoi) 

    train_iterator, test_iterator = torchtext.data.BucketIterator.splits(
                                (trn, tst),
                                batch_size = args.batch_size,
                                sort_key=lambda x: len(x.comment),
                                sort_within_batch=False)
    input_dim = len(TEXT.vocab)
    model = RNN(input_dim, 
            args.embedding_dim, 
            args.hidden_dim, 
            args.output_dim, 
            args.n_layers, 
            args.birnn, 
            args.dropout)
    
    print(model)
    #print("bidirectional: {} ".format(args.birnn))

    pretrained_embeddings = TEXT.vocab.vectors
    #print(pretrained_embeddings.shape)
    model.embedding.weight.data.copy_(pretrained_embeddings)

    #initial embedding layers not with pretained embeddings
    #torch.nn.init.xavier_normal_(model.embedding.weight.data)
    #torch.nn.init.kaiming_normal_(model.embedding.weight.data)

    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    model.embedding.weight.data[unk_idx] = torch.zeros(args.embedding_dim)
    model.embedding.weight.data[pad_idx] = torch.zeros(args.embedding_dim)
    print(model.embedding.weight.data)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    all_losses = []
    all_trainAcc = []

    for epoch in range(args.num_epochs):     
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        all_losses.append(train_loss)
        all_trainAcc.append(train_acc)
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% |')
    
    # figfile = args.output_dir + 'loss_epoches_{:02}.png'.format(args.num_epochs)
    # fig = plt.figure(figsize=(10, 8))
    # plt.plot(all_losses)
    # plt.title('training loss')
    # fig.savefig(figfile)

    # figfileAcc = args.output_dir + 'TrainAcc_epoches_{:02}.png'.format(args.num_epochs)
    # fig1 = plt.figure(figsize=(10, 8))
    # plt.plot(all_trainAcc)
    # plt.title('training Acc')
    # fig1.savefig(figfileAcc)

    #plot in one file
    figfile = args.output_dir + 'epoches_{:02}.png'.format(args.num_epochs)
    f = plt.figure(figsize=(10,6))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax1.plot(all_trainAcc)
    ax1.set_title("Train Accuracy")
    ax2.plot(all_losses, 'r:')
    ax2.set_title('Train Loss')
    f.savefig(figfile)

    #model evaluation
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in test_iterator:
            predictions = model(batch.comment).squeeze(1)
            loss = criterion(predictions, batch.label)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct = (rounded_preds == batch.label).float()            
            acc = correct.sum()/len(correct)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    test_loss = epoch_loss / len(test_iterator)
    test_acc = epoch_acc / len(test_iterator)
    
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    # run.log("Test accuracy", f'{test_acc*100:.2f}%')
    # run.log("Test loss", f'{test_loss:.3f}')
    modelfile = args.save_dir + 'model_acc_{:.3f}_epochs_{:02}.pt'.format(test_acc, args.num_epochs)
    vocabfile = args.save_dir + 'vocab.pt'
    torch.save(model, modelfile)
    torch.save(TEXT.vocab, vocabfile)

    comment = 'you really are sickening'
    comment1 = "Oh, you're gay? You should meet my friend Ann. She's gay, too!"
    comment2 ="Your name is so hard to pronounce"
    comment3 = 'my boss is crazy'
    comment4 = 'Why do you wear that?'
    comment5 = "As a woman, I know what you go through as a racial minority."
    comment6 = "You're transgender? Wow, you don't look like it at all"
    prediction(comment, vocabfile, modelfile)
    prediction(comment1, vocabfile, modelfile)
    prediction(comment2, vocabfile, modelfile)
    prediction(comment3, vocabfile, modelfile)
    prediction(comment4, vocabfile, modelfile)
    prediction(comment5, vocabfile, modelfile)
    prediction(comment6, vocabfile, modelfile)
    end_time = time.time()-begin_time
    print("Total execution time: ", format_timespan(end_time))

if __name__ == "__main__":
    main()
