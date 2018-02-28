import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable

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

class CharRNN(nn.Module):
  def __init__(self,embedding_size,h_size,num_layer=1,num_directions=1,cell_type=torch.nn.LSTM,loss_function=
        #http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        #http://pytorch.org/docs/master/nn.html#torch.nn.LSTM
        torch.nn.CrossEntropyLoss() ):
        super(CharRNN, self).__init__()
        self.embedding_size = embedding_size
        self.h_size = h_size
        self.num_layer = num_layer
        self.num_directions = num_directions
        assert self.num_directions<3
        if self.num_directions > 1:
          self.cell = cell_type(embedding_size,h_size,num_layer,bidirectional = True)
        else:
          self.cell = cell_type(embedding_size,h_size,num_layer)

        #ascii
        self.f1 = torch.nn.Linear(h_size,256)

        self.softmax = nn.Softmax()
        self.loss_function  = loss_function
        self.optimizer = optim.SGD(self.parameters(), lr=0.1)

  def onehotEncoding(self,id_list):
    # ascii biggerst is 255
    assert min(id_list)>=0
    ascii_max_dim = 256
    num_of_chars = len(id_list)
    onehot = Variable(torch.zeros(num_of_chars, ascii_max_dim ).scatter_(1, torch.LongTensor([[i] for i in id_list]),1))
    return onehot



    #https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
  def initialize_h0_c0(self,batch_size):
    #          h0,c0
    return  Variable(torch.zeros(self.num_layer * self.num_directions, batch_size, self.h_size),requires_grad=False),Variable(torch.zeros(self.num_layer * self.num_directions, batch_size, self.h_size),requires_grad=False)

    # X  : batches of X's (one hot/embedding) vectors (N , sequence_num )
    # y  : batches of y (N ,sequence_num )
  def forward(self,Xs,ys,predict):
    self.zero_grad()
    # LSTM
    #embedding Xs: (N , sequence_num,256 )

    Xs = [self.onehotEncoding(X)  for X in Xs]
    ys = [Variable(torch.LongTensor(y) ) for y in ys]




    prob_list = []
    predict_list = []

    total_loss = Variable(torch.FloatTensor([0.0])  , requires_grad=True)
    #必須要把轉換成3d dim的operation放在這裡
    #batch 裡面的sequence 一個一個的x拿出來用,所以batch size = 1
    for _X,y in zip(Xs,ys):  # loop through observations
                #          1     ,   N  , embedding_dim
                #  input (seq_len , batch, input_size)
      hiddens = self.initialize_h0_c0(1)
      X = _X.view( _X.size(0) , 1 , -1 )

      outputs = []
      #pytorch只能輸出最後一個time step的output,如果不用padding sequence的話
      for t in range(X.size(0)): # loop through time step to get all output[ts]
        Xt = X[t].view(1,1,-1)
        _out, hiddens =  self.cell(Xt, hiddens)

        #(1,output_dim)
        out = _out.view(-1)
        outputs.append(out)

      #(S,H)
      output_tensor = torch.stack(outputs,0)
      #(S,256)
      logits =  self.f1(output_tensor)

      if predict :
        # (S*256)
        prob =  self.softmax(logits)

        prob_list.append(prob)
        # (N)
        predict_y = torch.max(prob,1)[1]
        predict_list.append(predict_y)

      else:
        loss = self.loss_function(logits,y)
        total_loss = total_loss + loss


    if predict:
      return prob_list,predict_list

    average_loss=total_loss/len(Xs)
    return average_loss


  def backward(self,loss):
        loss.backward()
        self.optimizer.step()

  # input : context             list of token ids of begin of sentences
  #output string of generate
  def sample(self,context,eos=True,generate_len=100):
    #vectors (context_len,1,256)
    vectors = self.onehotEncoding(context).view(len(context),1,-1)

    hiddens = self.initialize_h0_c0(1)
    #context_vector(1,1,h_size)
    context_vector, hiddens =  self.cell(vectors, hiddens)
    print('context vector')
    print(context_vector.size())


    tokenids = []
    cnt = 0
    out = context_vector
    while  cnt<100 :
      if cnt == generate_len:
        break
      # (1*256)
      logit =  self.f1(out)
      prob =  self.softmax(logit)
      predict_token = torch.max(prob,1)[1]
      if predict_token  == 0:
        break
      tokenids.append(predict_token)
      cnt+=1
      out,hiddens = self.cell(out,hiddens)
    print('generate length')
    generate_sentence = [chr(t) for t in context] + [chr(t) for t in tokenids]
    generate_sentence = "".join(generate_sentence)
    print(len(generate_sentence))
    return generate_sentence






