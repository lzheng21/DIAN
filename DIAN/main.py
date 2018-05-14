from DIAN.data import *
from DIAN.models.BPR import BPR
from DIAN.models.FM import FM

data = Data(train_file='data/ml-100k/'+'train_users.dat', test_file='data/ml-100k/'+'test_users.dat')

model = FM(data,16,16,4,1024,{0:1,1:50},0.01,0.01)
model.train(6000, lr=0.001, optimizer='Adam')
model.predict()
