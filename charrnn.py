import torch

# step 1 : char --> id
# step 1.5 : split to batches (by period token)
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
def preprocessing(s):
    return  [ord(c) for c in s]

def text2Batches(text):
  return [ preprocessing(s)  for s in text.split(".") if len(s)>0]


def id2onehot(id_list):
  # ascii biggerst is 255
  assert min(id_list)>=0
  ascii_max_dim = 256
  num_of_chars = len(id_list)
  onehot = torch.zeros(num_of_chars, ascii_max_dim ).scatter_(1, torch.LongTensor([[i] for i in id_list]),1)
  return onehot





#sequence is 1 because we use loop for rnn implementation
def create_rnn_cell(embedding_size,h_size,num_layers=1,num_directions=1):
#http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
#http://pytorch.org/docs/master/nn.html#torch.nn.LSTM
  bidirectional  = False
  if num_directions >1:
    bidirectional  = True
  return torch.nn.LSTM(embedding_size, h_size , num_layers,bidirectional=bidirectional)

def create_references(text):
  l = [ord(c) for c in text]
  return l[1:]+[0]

def loss(zs,ys):
  #http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss
  loss = torch.nn.CrossEntropyLoss()(zs,ys)
  return loss


class TrainingManager():
  def __init__():
    pass

class CharRNN():
    def __init__(self,embedding_size,h_size,num_layer=1,num_directions=1,cell_type=torch.nn.LSTM,loss_function=torch.nn.CrossEntropyLoss() ):
        self.embedding_size = embedding_size
        self.h_size = h_size
        self.cell = cell_type(embedding_size,h_size)
        self.loss_function  = loss_function
        self.num_layer = num_layer
        self.num_directions = num_directions


    #https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    def initialize_h0_c0(batch_size,h_size,num_layers=1,num_directions=1):
    #          h0,c0
      return torch.zeros(num_layers * num_directions, batch_size, h_size),torch.zeros(num_layers * num_directions, batch_size, h_size)

    # X  : batches of X's (one hot/embedding) vectors
    # y  : batches of y
    def forward(self,Xs,ys):
      #LSTM
      #
      for X in Xs:
                                        #input (seq_len, batch, input_size)
        out, hiddens = self.cell(X, hiddens)


    def backward(self):
        pass
