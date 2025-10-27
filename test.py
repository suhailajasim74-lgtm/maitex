import pandas as pd
import numpy as np
df=pd.read_csv(r'C:\Users\user\Downloads\Predict Hair Fall.csv')
x=df.drop(['Hair_fall'],axis=1)
y=df['Hair_fall']
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)