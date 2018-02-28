from charrnn import CharRNN
from datautil import *
from train import *
import sys,torch

# https://discuss.pytorch.org/t/print-a-verbose-version-of-a-tensor/11201
PRINT_TO_FILE = False
torch.set_printoptions(threshold=5000)

def test_batches(s="abcdefg.hijklmn.wwaf."):
    batches = DataHandler(s).text2Batches()
    for X,y in batches:

        print("X:")
        print(X.size())
        print("y:")
        _y = y
        print(_y)

def test_predict(s="abcdefg.hijklmn.wwaf."):
    dh = DataHandler(s)

    dh.text2Batches()
    for batch in dh.getNextBatch():
        for observation in batch:
            Xs,ys = observation[0],observation[1]
            print("X:")
            print(Xs.size())
            print("y:")
            print(ys)

def test_forward(s="abcdefg.hijklmnxxx.wwaf."):
    dl = DataLoader()
    dl.load(s)
    spliter = DatasetSpliter(dl)
    spliter.split()
    trainingData = spliter.getTrainingData()
    bf = BatchFetcher(trainingData)

    model = CharRNN(embedding_size=256,h_size=512)
    tm = TrainingManager(500)
    s=model.sample([ord(c) for c in "ab"])
    print('generate : ')
    print(s)
   # tm.train(bf,model)



test_forward()
#id_batches = text2Batches("abcdefg.hijklmn.wwaf.")
#print(id_batches)
#batches_sequence=  [id2onehot(id_list) for id_list in id_batches]
#print(batches_sequence)
#print(loss(Variable(torch.Tensor([[0.25,0.25,0.5,0.5],[0.75,0.25,0,0]])),Variable(torch.LongTensor([3,0]))))




