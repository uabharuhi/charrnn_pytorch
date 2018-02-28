import torch
class TrainingManager():
  def __init__(self,epoch_num):
    self.epoch_num = epoch_num
  def train(self,fetcher,model):
    for i in range( self.epoch_num):
      loss_total = 0
      cnt  = 0
      for Xs,ys in  fetcher.getBatches():
        loss = model.forward(Xs,ys,False)
        #print(loss)
        loss_total+=loss
        model.backward(loss)
        cnt+=1
      loss_avg = loss_total/cnt
      print('loss at epoch:%d '%(i))
      print('print average loss')
      print(loss_avg)
    #save result
    print('results')
    torch.save(model,'./model')
