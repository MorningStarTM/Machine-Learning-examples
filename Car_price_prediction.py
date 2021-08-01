import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

ln = LinearRegression()
labelencoder = LabelEncoder()
df = pd.read_csv('car data.csv')
print(df.shape)
#features
##['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']

df_copy = df.copy()

#1-97 index for car names
df_copy.iloc[:,0] = labelencoder.fit_transform(df.iloc[:,0])

df_copy.iloc[:,5] = labelencoder.fit_transform(df.iloc[:,5])
df_copy.iloc[:,6] = labelencoder.fit_transform(df.iloc[:,6])
df_copy.iloc[:,7] = labelencoder.fit_transform(df.iloc[:,7])

del df_copy['Present_Price']
df_copy['Present_Price'] = df['Present_Price']

#X = df_copy[]
X = df_copy.iloc[:,df_copy.columns != 'Present_Price']
y = df_copy['Present_Price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

ln.fit(X_train,y_train)

y_pred = ln.predict(X_test)

print('MSE: ',metrics.mean_squared_error(y_test,y_pred))
print('MAE: ',metrics.mean_absolute_error(y_test,y_pred))
print('RMSE: ',np.asscalar(np.squeeze(metrics.mean_squared_error(y_test,y_pred))))
print('Score: ',ln.score(X_test,y_test))




