import pandas as pd
import numpy as np 
import pickle
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


#Load Data Set 
df = pd.read_csv('cleaned_recruitment.csv')

#Drop Unused Column 
df.dropna(axis=0, inplace=True)

#Split Feature and Target 
X = df.drop('Recruitment_Status', axis=1) #Feature
Y = df['Recruitment_Status'] #Target

#Imbalanced Data Handling
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, Y)

#Model and fit model 
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

#Save Model to disk 
pickle.dump(rfc, open('model.pkl','wb'))

#Load model 
model = pickle.load(open('model.pkl','rb')) 