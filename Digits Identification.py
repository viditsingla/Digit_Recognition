import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#%%

Load_data=load_digits()
Load_data.__sizeof__()
Load_data.items()
Load_data.keys()
Load_data.__init__()
Load_data
Load_data.__len__()
Load_data.values()
Load_data.data
Load_data.target
Load_data.target_names
Load_data.images



df_x=pd.DataFrame(Load_data.data)
df_x
df_y=pd.DataFrame(Load_data.target)
df_y

#Note that load_digits is a 'bunch' data type that has multiple data types within.
#load_digits.data is an array that contains the independent variables
#load_digits.target is a vector that contains the dependent variable


#%% Playing with df_x=load_digits (independent vars)
df_x.head(5)
df_x.corr()
df_x.count(axis=0)
df_x.index
df_x.isnull()
df_x.shape
df_x.describe()


#%% concatinating df_x and df_y
df=pd.concat([df_x,df_y],axis=1)
df.shape


#%% test train split
X_train, X_test, y_train, y_test = train_test_split(df_x,df_y,\
                                                    test_size=.27, random_state=0)


#%% Plain Logistic Regrssion
logreg=LogisticRegression()

logreg.fit(X_train, y_train)

pred_1=logreg.predict(X_test)

print("__"*50,"\n")
print('The accuracy of the Logistic Regression is',\
      accuracy_score(y_test,pred_1))
print("__"*50,"\n")

#Accuracy of prediction using logistic regression is 94.855%


#%% GridsearchCV for tuning hyperparameters
parameter_candidates = [
  {'C':10.0**np.arange(-2,3), \
   'penalty':['l1','l2'], 'class_weight': ['balanced']},
  {'gamma': [0.001, 0.0001], \
   'C':10.0**np.arange(-2,3), 'penalty':['l1','l2'], 'class_weight': ['balanced']},
]
# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=logreg, cv=10, param_grid=parameter_candidates, n_jobs=-1)
# Train the classifier on data1's feature and target data
clf.fit(X_train, y_train)
pred_1=clf.predict(X_test)



