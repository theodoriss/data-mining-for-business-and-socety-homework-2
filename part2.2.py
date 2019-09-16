#!/usr/bin/env python
# coding: utf-8

# In[1]:
   #wrong

import csv
from collections import defaultdict
import numpy as np
from time import time


# In[ ]:


tstart=time()


# In[2]:


user_item=defaultdict(list) # dict with key the user and value the list of items this user has interacted with
with open("Base_Set___UserID__ItemID__PART_2_2.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        #print (int(''.join(line[:1])))#user
        #print (int(''.join(line[1:])))#item
        #if int(''.join(line[:1])) not in user_item.keys():
            #user_item[int(''.join(line[:1]))]=(int(''.join(line[1:])))
        #else:
        user_item[int(''.join(line[:1]))].append(int(''.join(line[1:])))        
        #print(''.join(line[:1]),''.join(line[1:])) #user-item


# In[3]:


personal_vector=defaultdict(list) # create dict with key the index of item and as value a tuple with name of other item and probability
with open("ItemID__PersonalizedPageRank_Vector.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        i=0
        while i <=1679: # 1679 is the index number of the last item
            a=int(''.join(line[1:]).split(',')[i].strip('[').strip(' ('))#item
            b=float(''.join(line[1:]).split(',')[i+1].strip(' ').strip(')').strip(')]'))
            #print(a,b)
            personal_vector[''.join(line[:1])].append((a,b))
            i+=2


# In[4]:


class rec_system:
    
    def __init__(self,userid):
        self.userid=userid
        #self.rlist=rlist
        self.reccomendations=self.reccomendations()
        self.gt=self.ground_truth()
        self.r_precision=self.r_precision()
        
        





    def reccomendations(self):
        item_id=set()
        recs=[]
        #recomendations for user 1683
        for item in user_item[self.userid]:
            if str(item) in personal_vector.keys():
                for object in personal_vector[str(item)]:
                    if object[0]!=item and object[1]>=0.000330402869469384: # threshold value, explanation in report
                        if object[0] not in user_item[self.userid] and object[0] not in  item_id:
                            recs.append(object)
                            item_id.add(object[0])
        recs= sorted(recs, key = lambda x: x[1],reverse=True) # sort by probability. again, if we sort by index, our r precision
        # will be significantly higher
        recs_list=[x[0] for x in recs]
        return recs_list





    def ground_truth(self):# ground truth for user1
        gt=[]
        with open("Ground_Truth___UserID__ItemID__PART_2_2.tsv") as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for line in tsvreader:
                if ''.join(line[:1])==str(self.userid):
                    gt.append(int(''.join(line[1:])))
        return gt


            #print (''.join(line[:1]))#user
            #print (''.join(line[1:]))#item
            #print(''.join(line[:1]),''.join(line[1:])) #user-item
            
            
    def r_precision(self):
        gt_=self.gt
        lista=self.reccomendations[:len(gt_)]#take same numbers of reccommendations as ground truth of each user
        rp = list(set(gt_).intersection(set(lista)))
        if len(gt_)!=0:
            return len(rp)/len(gt_)


# In[5]:


all_r=[]
for i in range(1683,2626): # from 1st till last user
    a=rec_system(i)
    #print(a.r_precision)
    all_r.append(a.r_precision)


# In[7]:


sum(filter(None, all_r))/(len(all_r)-sum(x is  None for x in all_r)) # since for some users there is not ground truth,we will not 
# calulate r precision for them. the code above removes the nones both from the sum but also from the length of the all_r list,
# which is the list that contains the r precisions for all users


# In[6]:


all_r


# In[ ]:


tend=time()


# In[ ]:


tend-tstart

