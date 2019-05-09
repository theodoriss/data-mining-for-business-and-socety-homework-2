#!/usr/bin/env python
# coding: utf-8

# In[28]:


# we check how many cpu cores we have on this machine, in order to specify the number of cores in the cross validate command
# we can always use the value (-1) to use all cpu cores, but we are requested to specify the number of cores in an integer bigger than 1
from multiprocessing import cpu_count
print(cpu_count())


# In[30]:



import os
import pprint as pp
from surprise import Dataset
from surprise import Reader

from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering

from surprise.model_selection import KFold
from surprise.model_selection import cross_validate



# path to file
file_path = os.path.expanduser('Part1/dataset/ratings.csv')


reader = Reader(line_format='user item rating', sep=',', rating_scale=[1, 5], skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
print("Loading data completed")


# ## Part 1.1

# In[31]:



kf = KFold(n_splits=5, random_state=0) 
print("splitting into 5 folds done.")



c_algo = NormalPredictor()
cross_validate(c_algo, data, measures=['RMSE'], cv=kf, verbose=True)

c_algo = BaselineOnly()
cross_validate(c_algo, data, measures=['RMSE'], cv=kf, verbose=True)


# In[32]:


# ALgorithms:
# k-NN 

## 1 - KNNBasic 
c_algo = KNNBasic()
cross_validate(c_algo, data, measures=['RMSE'], cv=kf, n_jobs = 4,  verbose=True)

# KNNBasic  0.9516


# In[33]:


# 2. KNNWithMeans 


c_algo = KNNWithMeans()
cross_validate(c_algo, data, measures=['RMSE'], cv=kf, n_jobs = 4,  verbose=True)
 
# KNNWithMeans 0.9324


# In[34]:


# 3. KNNWithZScore


c_algo = KNNWithZScore()
cross_validate(c_algo, data, measures=['RMSE'], cv=kf, n_jobs = 4,  verbose=True)

# KNNWithZScore 0.9314


# In[35]:


# 4. KNNBaseline 0.9085 (better than benchmark)


c_algo = KNNBaseline()
cross_validate(c_algo, data, measures=['RMSE'], cv=kf, n_jobs = 4,  verbose=True)

# KNNBaseline 0.9085 (3rd)


# In[36]:


## Matrix Factorization-based algorithms 
   
# 5. SVD 

c_algo = SVD()
cross_validate(c_algo, data, measures=['RMSE'], cv=kf, n_jobs = 4,  verbose=True)

# SVD 0.9084 (2nd)



# In[37]:


# 6. SVDpp

c_algo = SVDpp()
cross_validate(c_algo, data, measures=['RMSE'], cv=kf, n_jobs = 4,  verbose=True)


#  0.8951 (1st)



# In[24]:


# 7. NMF  0.9359 

c_algo = NMF()
cross_validate(c_algo, data, measures=['RMSE'], cv=kf, n_jobs = 4,  verbose=True)

# 0.9359 


# In[38]:


# 8. SlopeOne 

c_algo = SlopeOne()
cross_validate(c_algo, data, measures=['RMSE'], cv=kf, n_jobs = 4,  verbose=True)

#SlopeOne 0.9230 


# In[39]:


# 9. CoClustering 

c_algo = CoClustering()
cross_validate(c_algo, data, measures=['RMSE'], cv=kf, n_jobs = 4,  verbose=True)


#CoClustering 0.9351


# ## Part 1.2

# In[ ]:


# importing libraries
from surprise.model_selection.search import GridSearchCV
from surprise import Dataset, Reader, KNNBaseline, SVD
from surprise.model_selection import KFold
from timeit import default_timer
import os 
from psutil import cpu_count


# In[ ]:


start = default_timer()

#grid of parameters taken into account
param_grid = {"n_factors":[50, 200, 700], "n_epochs":[50, 70, 150], "lr_all": [0.003, 0.008, 0.01], 
               "reg_all" : [0.0, 0.06, 0.08, 0.1]}

grid = GridSearchCV(SVD, param_grid = param_grid , measures = ["rmse"], cv = kf, n_jobs = 4 )
grid.fit(data)


stop = default_timer() 


# In[ ]:


# the total time of execution 
print("total time", stop - start)

# the optimal parameters found 
print("optimal parameters", grid.best_params)

# the associated root mean squared error 
print("root mean squared error", grid.best_score)

# the number of cpu cores 
print("the total number of physical CPU-cores is ", cpu_count(logical=False))
print("the total number of logical CPU-cores is ", cpu_count(logical=True))


# In[ ]:


start = default_timer()

#grid of parameters
grid_of_parameters = {
    'bsl_options': {'method': ['als', 'sgd'],
                              'reg': [1, 2]},
    'k': [20, 40, 80], 
    'min_k': [1, 5], 
    'sim_options': { 
        'user_based': [False, True],
        'name': ['msd','cosine'], 
        'min_support': [3], 
    }
}

grid = GridSearchCV(KNNBaseline, param_grid = grid_of_parameters , measures = ["rmse"], cv = kf, n_jobs = 4 )
grid.fit(data)


stop = default_timer() 


# In[ ]:


# the total time of execution 
print("total time", stop - start)

# the optimal parameters found 
print("optimal parameters", grid.best_params)

# the associated root mean squared error 
print("root mean squared error", grid.best_score)

# the number of cpu cores 
print("the total number of physical CPU-cores is ", cpu_count(logical=False))
print("the total number of logical CPU-cores is ", cpu_count(logical=True))

