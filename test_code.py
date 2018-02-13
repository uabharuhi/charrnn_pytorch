from charrnn import id2onehot, text2Batches
import sys,torch
torch.set_printoptions(threshold=5000)
# https://discuss.pytorch.org/t/print-a-verbose-version-of-a-tensor/11201
f = open('out.txt', 'w',encoding="utf-8")
sys.stdout = f
#print(id2onehot([3,3,4,5,8,7,9,0]))
id_batches = text2Batches("abcdefg.hijklmn.wwaf.")
#print(id_batches)
#batches_sequence=  [id2onehot(id_list) for id_list in id_batches]
#print(batches_sequence)
