
#####################################################  Phase -1  #######################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

########################################### 1. LOAD DATA SET ###########################################################
path = 'F:\\College\\AI_year_3\\Machine Learning\\Assigments\\Project\\archive\\insurance.csv'
df = pd.read_csv(path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print('--------------------------------------------------------------------------------------------------------------')
print('X IS :')
print(X)
print('--------------------------------------------------------------------------------------------------------------')
print('Y IS :')
print(y)
print('--------------------------------------------------------------------------------------------------------------')


#####################################################  Phase -2  #######################################################

############################################### Handle Empty cells #####################################################
# Check if the data frame contain null values
print(df.isnull().sum())
# Check completed df contains no null values
print('--------------------------------------------------------------------------------------------------------------')

################################################# Encoding #############################################################
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1, 4, 5])],
    remainder='passthrough'
)
X = ct.fit_transform(X)
# One hot encode the data

################################################# Feature Scaling ######################################################
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)

############################################### Train-test Split #######################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=True)

#####################################################  Phase -3  #######################################################
################################################ Applying Models #######################################################


# ### Multiple linear regression
from sklearn.linear_model import LinearRegression
linear_reg_model1 = LinearRegression()
linear_reg_model1.fit(X_train, y_train)
y_pred_LR = linear_reg_model1.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_LR.reshape(len(y_pred_LR), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])
print('--------------------------------------------------------------------------------------------------------------')


# ### Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
poly_reg_model2 = LinearRegression()
poly_reg_model2.fit(poly_reg.fit_transform(X_train), y_train)
y_pred_PR = poly_reg_model2.predict(poly_reg.fit_transform(X_test))
print(np.concatenate((y_pred_PR.reshape(len(y_pred_PR), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])
print('--------------------------------------------------------------------------------------------------------------')


# ### Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
decisionTree_reg_model3 = DecisionTreeRegressor(random_state=0)
decisionTree_reg_model3.fit(X_train, y_train)
y_pred_DTR = decisionTree_reg_model3.predict(X_test)
print(np.concatenate((y_pred_DTR.reshape(len(y_pred_DTR), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])
print('--------------------------------------------------------------------------------------------------------------')


# ### Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
randoForest_reg_model4 = RandomForestRegressor(n_estimators=300, random_state=0)
randoForest_reg_model4.fit(X_train, y_train)
y_pred_RFR = randoForest_reg_model4.predict(X_test)
print(np.concatenate((y_pred_RFR.reshape(len(y_pred_RFR), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])
print('--------------------------------------------------------------------------------------------------------------')


# ### Support Vector Regression
from sklearn.svm import SVR
supportVector_reg_model5 = SVR(kernel='linear', C=20, epsilon=0.01)
supportVector_reg_model5.fit(X_train, y_train)
y_pred_SVR = supportVector_reg_model5.predict(X_test)
print(np.concatenate((y_pred_SVR.reshape(len(y_pred_SVR), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])
print('--------------------------------------------------------------------------------------------------------------')


# ## Evaluating Performance
from sklearn.metrics import r2_score
print('Evaluating Performance:-')

# ### Multiple Linear Regression
print('Multiple Linear Regression:', r2_score(y_test, y_pred_LR))

# ### Polynomial Regression
print('Polynomial Regression:', r2_score(y_test, y_pred_PR))

# ### Decision Tree Regression
print('Decision Tree Regression:', r2_score(y_test, y_pred_DTR))

# ### Random Forest Regression
print('Random Forest Regression:', r2_score(y_test, y_pred_RFR))

# ### Support Vector Regression
print('Support Vector Regression:', r2_score(y_test, y_pred_SVR))

print('--------------------------------------------------------------------------------------------------------------')
