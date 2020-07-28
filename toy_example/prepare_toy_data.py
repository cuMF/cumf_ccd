# coding: utf-8

# In[1]:

#prepare netflix data as an input to to cuMF
#data should be in ./data/netflix/
#assume input is given in text format
#each line is like 
#"user_id item_id rating"
import os
import pandas as pd
from six.moves import urllib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy import sparse
from sklearn.cross_validation import train_test_split




# In[4]:

#file look like
'''
1::122::5::838985046
1::185::5::838983525
1::231::5::838983392
1::292::5::838983421
1::316::5::838983392
1::329::5::838983392
1::355::5::838984474
1::356::5::838983653
1::362::5::838984885
1::364::5::838983707
'''
m = 6040
n = 3952


# In[5]:
user,item,rating = np.loadtxt('test.ratings', delimiter=' ', dtype=np.float32,unpack=True)

# user,item,rating, ts = np.loadtxt('ml-10M100K/ratings.dat', delimiter='::', dtype=np.int32,unpack=True)
print (user)
print (item)
print (rating)
print (np.max(user))
print (np.max(item))
print (np.max(rating))
print (user.size)


# In[6]:

user_item = np.vstack((user, item))


# In[7]:

# user_item_train, user_item_test, rating_train, rating_test = train_test_split(user_item.T, rating, test_size=1000006, random_state=42)
# nnz_train = 9000048
nnz_test = 100020


# In[8]:

#for test data, we need COO format to calculate test RMSE
#1-based to 0-based
R_test_coo = coo_matrix((rating,(user[:] - 1,item[:] - 1)))
assert R_test_coo.nnz == nnz_test
R_test_coo.data.astype(np.float32).tofile('R_test_coo.data.bin')
R_test_coo.row.tofile('R_test_coo.row.bin')
R_test_coo.col.tofile('R_test_coo.col.bin')


# In[9]:

print (np.max(R_test_coo.data))
print (np.max(R_test_coo.row))
print (np.max(R_test_coo.col))
print (R_test_coo.data)
print (R_test_coo.row)
print (R_test_coo.col)


# In[10]:

test_data = np.fromfile('R_test_coo.data.bin',dtype=np.float32)
test_row = np.fromfile('R_test_coo.row.bin', dtype=np.int32)
test_col = np.fromfile('R_test_coo.col.bin',dtype=np.int32)
print (test_data[0:10])
print (test_row[0:10])
print (test_col[0:10])


# In[11]:

