

# import project dependencies
import numpy as np
import pandas as pd


# read dataset into notebook
df = pd.read_csv('house_price_data.csv')

df.head()

# split X and y
X = df.drop(['Price','Id'],1)
y = df['Price']

# lets not make the train test split,just train the whole dataset and then run a crossvalidation test

# create a model

# import and instatiate the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# fit the model
model.fit(X,y)

# check the score on test set
model.score(X,y)

# check the cross val score
from sklearn.model_selection import cross_val_score
cross_val_score(model,X,y)

# now save the model to the disk using pickle

# import the pickle dependencies
import pickle

# save the model
pickle.dump(model,open('model.pkl','wb'))

# load the model for confirmation
loaded_model = pickle.load(open('model.pkl','rb'))

# do an instance testing of the loaded model
loaded_model.predict([[1245.3,10,12,3,0,0]])

# the model is working fine

