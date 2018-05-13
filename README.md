# DIAN: A TensorFlow Library for Implicit Data Recommendation


### Installation
pip install git+git://github.com/lzheng21/DIAN.git

### Supported Models
Bayesian Personalized Ranking (https://arxiv.org/abs/1205.2618)

Factorization Machines (https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
### Dependencies
Python 3.5

TensorFlow 1.5.0

NumPy >= 1.4.0

### Input Data Format
####train file:
uid1 iid_1 iid_2 ...

uid2 iid_5 iid_3 ...

####test file:
uid1 iid_3 iid_4 ...

uid2 iid_7 iid_8 ...

### Usage
from load_data import *
from models.BPR import *

data = Data(train_file='data/ml-1m/'+'train_users.dat', test_file='data/ml-1m/'+'test_users.dat')

model = BPR(data, emb_dim=16, batch_size=1024)

model.train(200, lr=0.01, optimizer='Adam')

model.predict()

###Performance
    Model   Precision@3 NDCG@3  MAP
    BPR     0.27        0.26    0.23

