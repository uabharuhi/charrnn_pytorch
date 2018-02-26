import torch
from torch import nn

# step 1 : char --> id
# step 1.5 : split to batches (by period token) TODO: train val test
# step 2 : id --> vectors (1-hot or embedding)
# step 3 : create a rnn cell (one layer or  multi layer)  which input is  vectors of  step2
# step 4 : create references and  cross entropy
# step 5 :  fowrad and backpropagation of step 4
# step 5.5 :  create output function for testing
# step 6 : create training loop and hyperparameters
# step 7 : fine tuning on dev -data
# step 8 :
test_text = "Cultivate your hunger before you idealize.\
Motivate your anger to make them all realize.\
Climbing the mountain, never coming down.\
Break into the contents, never falling down.\
My knee is still shaking, like I was twelve,\
Sneaking out of the classroom, by the back door.\
A man railed at me twice though, but I didn't care.\
Waiting is wasting for people like me.\
Don't try to live so wise.\
Don't cry 'cause you're so right.\
Don't dry with fakes or fears,\
'Cause you will hate yourself in the end.\
Don't try to live so wise.\
Don't cry 'cause you're so right.\
Don't dry with fakes or fears,\
'Cause you will hate yourself in the end.\
You say, Dreams are dreams.\
I ain't gonna play the fool anymore.\
You say, Cause I still got my soul.\
Take your time, baby, your blood needs slowing down."

#class TextInputTransfer:
#    def __init__(self):
#        pass
#    def
#


#sequence is 1 because we use loop for rnn implementation
def create_rnn_cell(embedding_size,h_size,num_layers=1,num_directions=1):
#http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
#http://pytorch.org/docs/master/nn.html#torch.nn.LSTM
  bidirectional  = False
  if num_directions >1:
    bidirectional  = True
  return torch.nn.LSTM(embedding_size, h_size , num_layers,bidirectional=bidirectional)



def loss(zs,ys):
  #http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss
  loss = torch.nn.CrossEntropyLoss()(zs,ys)
  return loss



class DataHandler():
  def __init__(self,text):
    self.text = text

  # return batch_X (N,), batch_y ()
  def text2Batches(self):
    # split text by period
    text = self.text
    charID_batches =   [[ord(c) for c in s]  for s in text.split(".") if len(s)>0]
    print(charID_batches)
    #                 X              , y      
    batches =  [ (self.id2onehot(IDS), torch.LongTensor(IDS[1:]+[0])) for IDS in charID_batches]
    print(batches)
    return batches

  def id2onehot(self,id_list):
    # ascii biggerst is 255
    assert min(id_list)>=0
    ascii_max_dim = 256
    num_of_chars = len(id_list)
    onehot = torch.zeros(num_of_chars, ascii_max_dim ).scatter_(1, torch.LongTensor([[i] for i in id_list]),1)
    return onehot



class TrainingManager():
  def __init__(self,epoch_num):
    pass




class CharRNN():
    def __init__(self,embedding_size,h_size,num_layer=1,num_directions=1,cell_type=torch.nn.LSTM,loss_function=torch.nn.CrossEntropyLoss() ):
        
        self.embedding_size = embedding_size
        self.h_size = h_size
        self.num_layer = num_layer
        self.num_directions = num_directions
        if num_directions > 1:
          self.cell = cell_type(embedding_size,h_size,num_layer,bidirectional = True)
        else:
          self.cell = cell_type(embedding_size,h_size,num_layer)


        self.softmax = nn.Softmax(dim=2)
        self.loss_function  = loss_function
  





    #https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    def initialize_h0_c0(self,batch_size):
    #          h0,c0
      return  torch.zeros(self.num_layers * self.num_directions, batch_size, self.h_size),torch.zeros(self.num_layers * self.num_directions, batch_size, self.h_size)

    # X  : batches of X's (one hot/embedding) vectors (N , embedding dim )
    # y  : batches of y (N , )
    def forward(self,Xs,ys,predict=False):
      # LSTM
      
      #outputs = []
      # turn X to 3d (1,N,embedding_dim)
      print('initial state')
      batch_size = len(Xs)
      hiddens = self.initialize_h0_c0(1) # not require grad so it is a tensor , loop so batch size is 1

      total_loss = 0.0
      #必須要把轉換成3d dim的operation放在這裡
      for _X in Xs:      #          1     ,   N  , embedding_dim
                         # input (seq_len , batch, input_size)
        X = _X.view(1,_X.size(0),-1)               
        out, hiddens =  self.cell(X, hiddens)

        #nn.Softmax()()
        #outputs.append(out)
        if predict :
          prob =   self.softmax(X)
        else:
          pass
          
      #TODO : schedule sampling ?

      #






    def backward(self):
        pass
