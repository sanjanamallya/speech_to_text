
class pBLSTM(nn.Module):

  def __init__(self, input_size , hidden_size):
      super(pBLSTM, self).__init__()
      self.blstm = nn.LSTM(input_size = input_size*2, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first = True)
    
  def forward(self,x, lens):
    '''
    :param x :(N,T, H1) input to the pBLSTM h
    :return output: (N,T,H) encoded sequence from pyramidal Bi-LSTM 
    '''
    batch_size = x.data.size(0)
    timestep = x.data.size(1)
    feature_dim = x.data.size(2)

    if timestep % 2 != 0:
        x = x[:, :-1, :]
        timestep -= 1

    x = x.view(batch_size, timestep//2, 2, feature_dim)
    x = torch.mean(x, 2)
    lens = [length//2 for length in lens]
    inp = utils.rnn.pack_padded_sequence(x, lens, batch_first= True, enforce_sorted=False)  
    output, hidden = self.blstm(inp)
    op, new_lens = utils.rnn.pad_packed_sequence(output, batch_first= True)
    return op, hidden, new_lens

class Encoder(nn.Module):
    
  def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
    super(Encoder, self).__init__()
    self.lstm = nn.LSTM(input_size=input_dim,hidden_size=hidden_dim,num_layers=1,bidirectional=True)
    
    #Define the blocks of pBLSTMs
    self.pblstm1 = pBLSTM(input_size = hidden_dim , hidden_size=hidden_dim)
    self.pblstm2 = pBLSTM(input_size = hidden_dim , hidden_size=hidden_dim)
    self.pblstm3 = pBLSTM(input_size = hidden_dim , hidden_size=hidden_dim)

    self.key_network = nn.Linear(hidden_dim*2, value_size)
    self.value_network = nn.Linear(hidden_dim*2, key_size)
  
  def forward(self,x, lens):
        

    rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first= True, enforce_sorted=False)
    outputs, _ = self.lstm(rnn_inp)
    op, new_lens = utils.rnn.pad_packed_sequence(outputs, batch_first= True)
    output1, _, lens1 = self.pblstm1(op, new_lens)
    output2, _, lens2 = self.pblstm2(output1, lens1)
    output3, _, lens3 = self.pblstm3(output2, lens2)

    keys = self.key_network(output3)
    value = self.value_network(output3) 
    
    return keys, value, lens3

class Decoder(nn.Module):
    
  def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128,  isAttended=False):
    super(Decoder, self).__init__()
    
    self.embedding = nn.Embedding(vocab_size, hidden_dim)
    self.hidden_size = hidden_dim
    
    self.lstm1 = nn.LSTMCell(input_size=hidden_dim+value_size, hidden_size=hidden_dim)
    self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)
    
    self.isAttended = isAttended
    if(isAttended):
      self.attention = Attention()
    self.character_prob = nn.Linear(key_size+value_size,vocab_size)

  def forward(self, key, values, lens, teacher_force_cutoff, text=None, train=True):
    '''
    :param key :(T,N,key_size) Output of the Encoder Key projection layer
    :param values: (T,N,value_size) Output of the Encoder Value projection layer
    :param text: (N,text_len) Batch input of text with text_length
    :param train: Train or eval mode
    :return predictions: Returns the character perdiction probability 
    '''
       
    batch_size = key.shape[0]
    
    if(train):
      max_len =  text.shape[1]
      embeddings = self.embedding(text)
    
    else:
      max_len = 250 #hyperparameter
    
    predictions = []
    hidden_states = [None, None]
    prediction = torch.zeros(batch_size,1).to(device)

#     Values # B x T x128
    context = values[:,0,:]

    for i in range(max_len):
        '''
        Implementation of Gumble noise and teacher forcing techniques
        '''
        teacher_force = True if np.random.random_sample() < teacher_force_cutoff else False

        if(train):
            if teacher_force:
                char_embed = embeddings[:,i,:]
            else:                
                if teacher_force_cutoff<=0.6:
                    noise_param = 0.1
                else:
                    noise_param = 0.2
                prediction = Gumbel(prediction.to('cpu'), torch.tensor([noise_param])).sample().to(device)
                char_embed = self.embedding(prediction.argmax(dim=-1))


        else:
            char_embed = self.embedding(prediction.argmax(dim=-1))

        inp = torch.cat([char_embed,context], dim=1)
        hidden_states[0] = self.lstm1(inp,hidden_states[0])

        inp_2 = hidden_states[0][0]
        hidden_states[1] = self.lstm2(inp_2,hidden_states[1])

        output = hidden_states[1][0]
        context, op_viz = self.attention(output, key, values, lens)  # lens here is lens3 from encoder

        prediction = self.character_prob(torch.cat([output, context], dim=1))
        predictions.append(prediction.unsqueeze(1))

    return torch.cat(predictions, dim=1)

class Attention(nn.Module):
  def __init__(self):
    super(Attention, self).__init__()
  def forward(self, query, key, value, lens):
    '''
    :param query :(N,context_size) Query is the output of LSTMCell from Decoder
    :param key: (N,T,key_size) Key Projection from Encoder per time step
    :param value: (N,value_size) Value Projection from Encoder per time step
    :return output: Attended Context
    :return attention_mask: Attention mask that can be plotted  
    '''

    # Compute (N, T) attention logits. "bmm" stands for "batch matrix multiplication".
    # Input/output shape of bmm: (N, T, H), (N, H, 1) -> (N, T, 1)
    attention = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
    # Create an (N, T) boolean mask for all padding positions
    # Make use of broadcasting: (1, T), (N, 1) -> (N, T)
    mask = torch.arange(key.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
    mask= mask.to(device)
    attention.masked_fill_(mask, -1e9)
    attention = nn.functional.softmax(attention, dim=1)
    # Input/output shape of bmm: (N, 1, T), (N, T, H) -> (N, 1, H)
    out = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
    return out, attention

class Seq2Seq(nn.Module):
  def __init__(self,input_dim,vocab_size,hidden_dim,value_size=128, key_size=128,isAttended=False):
    super(Seq2Seq,self).__init__()
    self.encoder = Encoder(input_dim, hidden_dim)
    self.decoder = Decoder(vocab_size, hidden_dim, isAttended= True)

  def forward(self,speech_input, speech_len, teacher_forcing_ratio, text_input_preds = None, train=True):
    key, value, lens = self.encoder(speech_input, speech_len)
    if(train):
      predictions = self.decoder(key, value, lens, teacher_forcing_ratio, text_input_preds)
    else:
      predictions = self.decoder(key, value, lens, teacher_forcing_ratio, text=None, train=False)
    return predictions