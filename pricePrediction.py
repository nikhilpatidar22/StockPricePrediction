import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model as lm
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('.\\RELIANCE.NS.csv')
df
df['Open-Close']=df['Close']-df['Open']
df['tommrow']=df['Close'].shift(-1)
df.head()


df['target']=(df['Close']<df['tommrow']).astype(int)
df=df.fillna(method="ffill")
df
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(df[['Open','Low','High','Close','Open-Close']],df.tommrow,test_size=0.20)

x_train
reg=lm.LinearRegression()
reg.fit(x_train,y_train)
reg.coef_
reg.predict(x_test)
reg.score(x_test,y_test)