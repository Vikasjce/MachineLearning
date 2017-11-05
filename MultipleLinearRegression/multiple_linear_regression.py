# Multiple Linear regression

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the Dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4]

# Taking care of missing data 
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
#imputer = imputer.fit(X[:,1:3])
#X[:,1:3]=imputer.transform(X[:,1:3])

# Encoding a Categorical Variable 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]

# Splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Feauture Scaling


# Fitting Multi Linear regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the test results 
y_pred = regressor.predict(X_test)

# Building Backword elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
