# turn raw data into  list of X,y or just [(X)....] in unsupervised
class DataLoader():
  def __init__(self):
    self.data = None

  # return data [(X,y).....] or just [(X)....] in unsupervised
  def load(self,text):
    # split text by period
    charID_batches =   [[ord(c) for c in s]  for s in text.split(".") if len(s)>0]
    self.data = [ (IDS,IDS[1:]+[0]) for IDS in charID_batches]
    return self.data

#split data to train,val,test
# train/val/test: (Xs,ys)
class DatasetSpliter():
  def __init__(self,dataloader): # can add  other split arguemnts in constuctor then call split
    self.dataloader = dataloader

  def split(self):
    self.trainning_data =  self.dataloader.data

  def getTrainingData(self):
    return self.trainning_data

# dataset list of data : list of  X,y or just [(X)....] in unsupervised
class BatchFetcher():
  def __init__(self,batches,batch_size=2):
    self.currentIndex = 0
    self.Xs = []
    self.ys = []
    self.batch_size = batch_size
    self.data_num  = len(batches)
    for X,y in batches:
      self.Xs.append(X)
      self.ys.append(y)

  # generator :  get batches of data
  def getBatches(self):
    assert  self.Xs  is not None
    assert  self.ys  is not None

    while True:
      if self.currentIndex < self.data_num:

        nextIndex = self.currentIndex + self.batch_size
        if nextIndex >= self.data_num:
          yield self.Xs[self.currentIndex:self.data_num],self.ys[self.currentIndex:self.data_num]
          break

        ret_Xs,ret_ys = self.Xs[self.currentIndex:nextIndex],self.ys[self.currentIndex:nextIndex]
        self.currentIndex = nextIndex
        yield ret_Xs,ret_ys

  def reset(self):
    self.currentIndex = 0
