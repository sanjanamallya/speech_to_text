import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
import pickle as pk
from torch.utils.data import DataLoader, Dataset 
from torch.distributions.gumbel import Gumbel
import time
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from model import *
from utils import *

# DATASET
class Speech2Text_Dataset(Dataset):
        def __init__(self, speech, text=None, train=True):
            self.speech = speech
            self.train = train
            if(text is not None):
                  self.text = text
                              
        def __len__(self):
            return self.speech.shape[0]
        def __getitem__(self, index):
            if(self.train):
                # CHECK
                # self.text[index][0:-1] -> decoder
                #preds (sos to last char) loss(first char to eos)
                return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index][:-1]), torch.tensor(self.text[index][1::])

            else:
                return torch.tensor(self.speech[index].astype(np.float32))

#COLLATE FN
def collate_train(batch_data):
    '''
    returns padded speech and text data, and length of 
    utterance and transcript from this function 
    '''
    # CHECK
   
    inputs, targets_preds, targets_loss = zip(*batch_data)
    
    inputs = [inputs[i] for i in range(len(inputs))]
    input_seq_lens = torch.LongTensor([len(seq) for seq in inputs])
    inputs= torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)

    input_preds_seqlen = torch.LongTensor([len(seq) for seq in targets_preds])
    targets_preds=  torch.nn.utils.rnn.pad_sequence(targets_preds, batch_first=True, padding_value=0)

    input_loss_seqlen = torch.LongTensor([len(seq) for seq in targets_loss])
    targets_loss=  torch.nn.utils.rnn.pad_sequence(targets_loss, batch_first=True, padding_value=0)

    return inputs, targets_preds, targets_loss, input_seq_lens, input_preds_seqlen, input_loss_seqlen 

def collate_test(batch_data):
    '''
    Complete this function.
    I usually return padded speech and length of 
    utterance from this function 
    '''

    inputs = batch_data
    inputs = [inputs[i] for i in range(len(inputs))]
    input_seq_lens = torch.LongTensor([len(seq) for seq in inputs])
    inputs= torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)

    return inputs, input_seq_lens


def train(model,train_loader, teacher_forcing_ratio, criterion, optimizer):    
    loss_sum = 0

    for (batch_num, collate_output) in enumerate(tqdm(train_loader)):
          with torch.autograd.set_detect_anomaly(True):
            #LOOK
            optimizer.zero_grad()
            inputs, targets_preds, targets_loss, input_seq_lens, input_preds_seqlen, input_loss_seqlen = collate_output

            inputs = inputs.to(device)
            targets_preds = targets_preds.to(device)
            targets_loss = targets_loss.to(device)
            input_seq_lens = input_seq_lens.to(device)
            input_preds_seqlen = input_loss_seqlen.to(device)
            input_loss_seqlen = input_loss_seqlen.to(device)                   
            
            predictions = model(inputs, input_seq_lens , teacher_forcing_ratio, targets_preds)
            
            if  batch_num % 200 == 1:
                for i in range(predictions.size(0)):
                    string1 = ""
                    string2 = ""
                    curr_seq1 = predictions[i,0:input_preds_seqlen[i], :]
                    curr_seq1= torch.argmax(curr_seq1, dim=1)

                    curr_seq2 = targets_preds[i,0:input_preds_seqlen[i]]
     
                    string1 = "".join(idx_to_letter[int(char)] for char in curr_seq1)
                    string2 = "".join(idx_to_letter[int(char)] for char in curr_seq2)
                    print(string1, string2)
                    print("/n")
           
            mask = torch.zeros((predictions.size(1), predictions.size(0))).to(device)
            for idx, length in enumerate(input_loss_seqlen):
                    mask[:length,idx] = 1 #(T,B)
            mask = mask.T.contiguous().view(-1).to(device)
            
            predictions = predictions.contiguous().view(-1, predictions.size(-1)) # BxTx35
            targets_loss = targets_loss.contiguous().view(-1)

            loss = criterion(predictions,targets_loss)
            masked_loss = torch.sum(loss*mask)
            masked_loss.backward()
        
            torch.nn.utils.clip_grad_norm(model.parameters(), 2)
            optimizer.step()
            
            current_loss = float(masked_loss.item())/int(torch.sum(mask).item())
            if  batch_num % 5 == 0:
                print('train_loss', current_loss)
                
            del inputs, targets_preds, targets_loss, input_seq_lens, input_preds_seqlen, input_loss_seqlen, predictions

    return current_loss



def val(model,val_loader, teacher_forcing_ratio = 0):  
    
    decoded = list()
    loss_sum = 0

    print("---------------- VAL ------------", len(val_loader))

    for (batch_num, collate_output) in enumerate(tqdm(val_loader)):
          with torch.autograd.set_detect_anomaly(True):
                
            inputs, targets_preds, targets_loss, input_seq_lens, input_preds_seqlen, input_loss_seqlen = collate_output
            inputs = inputs.to(device)
            targets_preds = targets_preds.to(device)
            targets_loss = targets_loss.to(device)
            input_seq_lens = input_seq_lens.to(device)
            input_preds_seqlen = input_loss_seqlen.to(device)
            input_loss_seqlen = input_loss_seqlen.to(device)      
            
            predictions = model(inputs, input_seq_lens , teacher_forcing_ratio , None, False)
            if  batch_num % 2 == 1:
                for i in range(predictions.size(0)):
                    string1 = ""
                    string2 = ""
                    curr_seq1 = predictions[i,0:input_preds_seqlen[i], :]
                    curr_seq2 = targets_preds[i,0:input_preds_seqlen[i]]
                    curr_seq1= torch.argmax(curr_seq1, dim=1)
                    string1 = "".join(idx_to_letter[int(char)] for char in curr_seq1)
                    string2 = "".join(idx_to_letter[int(char)] for char in curr_seq2)
                    print("Model Prediction", string1)
                    print("Gold", string2)
                    test_string = index_to_letter(torch.argmax(predictions[i,:,:], dim = 1))
                    print("Stopping at eos",test_string)
                    decoded.append(test_string)
                    

    return decoded

def test(model,test_loader, teacher_forcing_ratio = 0):    
    loss_sum = 0

    decoded = list()
    print("---------------- TEST ------------", len(test_loader))
    for (batch_num, collate_output) in enumerate(tqdm(test_loader)):
          with torch.autograd.set_detect_anomaly(True):
                       
            #LOOK
            inputs, input_seq_lens = collate_output
            inputs = inputs.to(device)
            input_seq_lens = input_seq_lens.to(device)
            
            predictions = model(inputs, input_seq_lens , teacher_forcing_ratio , None, False)
            for i in range(predictions.size(0)):
                test_string = index_to_letter(torch.argmax(predictions[i,:,:], dim = 1))
                print("Test string", test_string)
                decoded.append(test_string)
    return decoded

def main():
    

    speech_train = np.load('train_new.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load('dev_new.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load('test_new.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('./train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load('./dev_transcripts.npy', allow_pickle=True,encoding='bytes')
    print("Data Loading Sucessful.....")


    model = Seq2Seq(input_dim=40,vocab_size=len(letter_list),hidden_dim= 512)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction = 'none').to(device)
    num_epochs = 27

    character_text_train =  transform_letter_to_index(transcript_train)
    character_text_valid = transform_letter_to_index(transcript_valid)
    print("Transformed data sucessfully.....")

    Speech2Text_train_Dataset = Speech2Text_Dataset(speech_train, character_text_train)
    Speech2Text_valid_Dataset = Speech2Text_Dataset(speech_valid, character_text_valid)
    Speech2Text_test_Dataset = Speech2Text_Dataset(speech_test, None, False)

    train_loader = DataLoader(Speech2Text_train_Dataset, batch_size= 32, shuffle=True, collate_fn=collate_train)
    val_loader = DataLoader(Speech2Text_valid_Dataset, batch_size= 32, shuffle= False, collate_fn=collate_train)


    teacher_forcing_ratio = 0.9         

    for epoch in range(num_epochs):
        if epoch % 3 ==0:
            teacher_forcing_ratio -= 0.1     
        if teacher_forcing_ratio <0.6:
            teacher_forcing_ratio = 0.6
        train_loss = train(model, train_loader, teacher_forcing_ratio, criterion, optimizer)
        path = "exp4_epoch"+str(epoch)
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss}, path)
        _ = val(model, val_loader)

    test_loader = DataLoader(Speech2Text_test_Dataset, batch_size= 64, shuffle= False, collate_fn=collate_test)
    decoded = test(model,test_loader)

    #Save results into csv file
    np.savetxt("results_exp.csv", decoded, delimiter=",", fmt='%s', header = "Predicted")

if __name__ == "__main__":
    main()
