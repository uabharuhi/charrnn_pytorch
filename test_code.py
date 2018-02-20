from charrnn import id2onehot, text2Batches,create_references,loss
import sys,torch


PRINT_TO_FILE = False

torch.set_printoptions(threshold=5000)
# https://discuss.pytorch.org/t/print-a-verbose-version-of-a-tensor/11201
if PRINT_TO_FILE:
    f = open('out.txt', 'w',encoding="utf-8")
    sys.stdout = f
#print(id2onehot([3,3,4,5,8,7,9,0]))
#id_batches = text2Batches("abcdefg.hijklmn.wwaf.")
#print(id_batches)
#batches_sequence=  [id2onehot(id_list) for id_list in id_batches]
#print(batches_sequence)
#print(loss(Variable(torch.Tensor([[0.25,0.25,0.5,0.5],[0.75,0.25,0,0]])),Variable(torch.LongTensor([3,0]))))

