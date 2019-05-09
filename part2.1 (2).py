#!/usr/bin/env python
# coding: utf-8

# In[12]:


import networkx as nx
from networkx.algorithms import bipartite
import scipy.sparse
import numpy as np
from scipy.sparse import csr_matrix
import csv
import matplotlib.pyplot as plt
from time import time
import matplotlib.pyplot as plt


# In[13]:


B = nx.Graph()


# In[14]:


with open("User_Item_BIPARTITE_GRAPH___UserID__ItemID.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        B.add_nodes_from([''.join(line[:1])], bipartite=0)
        B.add_nodes_from([''.join(line[1:])], bipartite=1)
        B.add_edge(''.join(line[:1]),''.join(line[1:]))
        #print (int(''.join(line[:1])))#user
       # print (int(''.join(line[1:])))#item
        
        #print(''.join(line[:1]),''.join(line[1:])) #user-item


# In[15]:


G = bipartite.generic_weighted_projected_graph(B,[n for n, d in B.nodes(data=True) if d['bipartite']==1])


# In[16]:


M = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(),weight='weight' ,dtype=float)


# In[17]:


Mnorm=csr_matrix(M.T/M.sum(axis=1).T)


# In[275]:


class rec_system:
    
    def __init__(self,userid):
        self.userid=userid
        self.topic=self.topic()
        #self.rlist=rlist
        self.pagerank=self.pagerank()
        self.reccomendations=self.reccomendations()
        self.gt=self.ground_truth()
        self.r_precision=self.r_precision()
        #self.self=self
        
        
    def topic(self):# find all the items that the current user has interacted with
        lst=[]
        for user,item in B.edges(str(self.userid)):
            lst.append(item)
        return lst





    def pagerank(self): # dumping factor is 0.1
        r = scipy.repeat(1.0 / len(G.nodes()), len(G.nodes()))
        # for user 1683
        rold=r
        r=0.9*Mnorm*r # probability based on transition matrix for each node
        iterations=1
        while scipy.absolute(r - rold).sum() > 0.000000001: # while loop till probabilities converge. also the sum of r vector should be 
            #equal to 1, something that we can confirm after testing.
            iterations+=1
            rold=r
            #print(iterations)
            r=0.1*Mnorm*r
            #print(r.shape)
            #for item in G.nodes():
            for index in range(len(r)):
                if str(index) in self.topic:
                    r[index]+=0.1*1/len(self.topic) # add the teleport probability to each node in  the topic
        return r


    def reccomendations(self):
        rec_list=[]#testlist
        i=0
        r = self.pagerank
        for item in np.argsort(r)[::-1][:len(self.topic)*2]: # take a lot of reccommendations, twich the size of each user-topic
            # and we do that because we expect the majority of top pagerank results to be items that the user has already interacted with
            if str(list(G.nodes())[item]) not in self.topic:# we cant recommend an item that the user has already interacted with
                #print(str(list(G.nodes())[item]),r[item])
                rec_list.append(int((list(G.nodes())[item])))
        return rec_list



    def ground_truth(self):# ground truth for each user
        gt=[]
        with open("Ground_Truth___UserID__ItemID.tsv") as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for line in tsvreader:
                if ''.join(line[:1])==str(self.userid):
                    gt.append(int(''.join(line[1:])))
        return gt


            #print (''.join(line[:1]))#user
            #print (''.join(line[1:]))#item
            #print(''.join(line[:1]),''.join(line[1:])) #user-item
            
            
    def r_precision(self): # returns r_precision for the specific input user
        gt_=self.gt
        lista=self.reccomendations[:len(gt_)]
        rp = list(set(gt_).intersection(set(lista)))
        if len(gt_)!=0:
            if gt_:
            #self.append(len(rp)/len(gt_))
            #self=len(rp)/len(gt_)
                return len(rp)/len(gt_)


# In[ ]:


all_r=[]
for i in range(1683,2626): # from 1st till last user
    a=rec_system(i)
    #print(a.r_precision)
    all_r.append(a.r_precision)


# In[286]:


sum(filter(None, all_r))/(len(all_r)-sum(x is  None for x in all_r)) # since for some users there is not ground truth,we will not 
# calulate r precision for them. the code above removes the nones both from the sum but also from the length of the all_r list,
# which is the list that contains the r precisions for all users


# In[ ]:


#we found out that if we sort our reccomendations by item index,instad of probabilities, the r precision is much bigger
# around 0.71 for user 1683. We didnt implement it but perhaps we should, since it looks pointless to compare so small numbers
# like the probabilities of each item in our case, and perhaps their differences are insignificant from a statistics aspect.


# In[293]:


all_r


# In[295]:


fig = plt.figure()
plt.hist([x for x in all_r if x is not None])
plt.title('histogram of r precision for number of users')
plt.xlabel('r_precision')
plt.ylabel('number of users')
fig.savefig('plot.png')


# In[ ]:




