from DIAN.load_data import *
from DIAN.models.BPR import *

data = Data(train_file='data/'+'train_users.dat', test_file='data/'+'test_users.dat')
model = BPR(data, emb_dim=64, batch_size=1024,lambda_u=0.02,lambda_v=0.02)
model.train(6000, lr=0.01, optimizer='Adam')
model.predict()

