import pandas as pd
import numpy as np
import os 

import pickle 

from sklearn.ensemble import RandomFOrestCLlassifier 

train_data= pd.read_csv("data/processed/train_processed.csv")
xtrain=train_data.iloc[:,0:-1].values
ytrain=train_data.iloc[:,-1].values

clf = RandomFOrestCLlassifier()
clf.fit(xtrain,ytrain)
pickle.dump(clf,open("model.pkl","wb"))
