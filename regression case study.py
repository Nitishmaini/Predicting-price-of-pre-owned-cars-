import pandas as pd
import numpy as np
import seaborn as sns

# set the dimensions for plots
sns.set(rc={'figure.figsize':(11.7,8.27)})

#. Reading Csv File
cars_data=pd.read_csv('cars_sampled.csv')
cars=cars_data.copy()

cars.info()

cars.describe()
pd.set_option('display.float_format',lambda x: '%.3f' % x)
cars_des=cars.describe()

# Drop unwanted columns
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col, axis=1)

# removing duplicate records
cars.drop_duplicates(keep='first', inplace=True) # 470 duplicate

# Data cleaning
cars.isnull().sum()

# variable yearofregistration
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
# we can see that future values are there which is not possible and year like 1000 1255 which is also not possible so we have to remove them all
sum(cars['yearOfRegistration']>2019)
sum(cars['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration',y='price',data=cars,scatter=True)
# we can clearly see that our graph doesnt making any sense so we have to remove faulty data
# working range of yearOfRegistration is 1950 to 2019

pricewise=cars['price'].value_counts().sort_index()
sns.distplot(cars['price']) # we can see there are lot of entry under 0 price category so we have to work on
cars['price'].describe() # the difference between mean and median is very high that means there are lot of outliers and std is also very high 
sns.boxplot(y=cars['price']) # we are not able to see the box because of lot of outliers so we have to find working range
sum(cars['price']>150000)
sum(cars['price']<100)
# we can work in this range 100 to 150000

# variable powerPS
power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])# we can see there are lot of outliers
cars['powerPS'].describe()
# std is high not ignorable difference between median and mean
sns.boxplot(y=cars['powerPS']) # we are not able to see box because of outliers
sns.regplot(x='powerPS',y='price',scatter=True,data=cars,fit_reg=False)
# regression plot is not making any sense we have to clean the data to see the relationship
sum(cars['powerPS']>500) # 5565
sum(cars['powerPS']<10)  # 115
# so we can take range from 10 to 500

# Working range of data
cars=cars[
    (cars.yearOfRegistration<=2019) 
          & (cars.yearOfRegistration>=1950) 
          & (cars.price>=100)
          & (cars.price<=150000) 
          & (cars.powerPS>=10) 
          & (cars.powerPS<=500)]
# almost 6700 records dropped
# Find the age of the vehicle using yearOfRegistration and monthOfRegistration
cars['monthOfRegistration']/=12
cars['Age']=(2019-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age'].describe()
# our std deviation is not high means our data is not diverse and difference between mean and median is very less that is good thing
cars['Age']=round(cars['Age'],2)

# Drop yearOfRegistration and monthOfRegistration
cars=cars.drop(columns=['monthOfRegistration','yearOfRegistration'], axis=1)

sns.distplot(cars['Age']) # Now we can fit density curve over it
sns.boxplot(cars['Age']) # this plot is better than previous plot but still there are some outliers

sns.distplot(cars['price']) # Now we can fit density curve over it
sns.boxplot(cars['price']) # this plot is better than previous plot but still there are some outliers

sns.distplot(cars['powerPS']) # Now we can fit density curve over it
sns.boxplot(cars['powerPS']) # this plot is better than previous plot but still there are some outliers

# visulizing parameter after narrowing work range
sns.regplot(x='Age',y='price',scatter=True,fit_reg=False, data=cars)
 # cars price higher are newer
 # With increase in age price dereased 
 # but some of the cars having high price even the age is also high we call them vintage cars

sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False, data=cars) 
# with increase in power price is also increasing

cars.info()

# we have done with numerical variable no categorical variable
# variable seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='counts',normalize=True) # 0% cars is from commercial seller
sns.countplot(x='seller',data=cars)
# only one is having commercial seller all other are private
# so insiginficant variable

# variable offertype
cars['offerType'].value_counts()
sns.countplot(x='offerType',data=cars)
# only one type of offer so insignificant variable

# variable abtest
cars['abtest'].value_counts()
sns.countplot(x='abtest',data=cars)
pd.crosstab(cars['abtest'],columns='counts',normalize=True)
# Equally distributed
sns.boxplot(x='abtest',y='price',data=cars)
# For every price value there is almost 50-50 distribution
# So it does not effect price so insignificant variable

# variable vehicleType
cars['vehicleType'].value_counts()
sns.countplot(x='vehicleType',data=cars)
pd.crosstab(cars['vehicleType'],columns='counts',normalize=True)
#Limousine, small car, station wagon have max freq
sns.boxplot(x='vehicleType',y='price',data=cars)
# we can see vehicle type is affecting price
# so imp variable

# variable gearbox
cars['gearbox'].value_counts()
sns.countplot(x='gearbox',data=cars)
pd.crosstab(cars['gearbox'],columns='counts',normalize=True)
sns.boxplot(x='gearbox',y='price',data=cars)
# we can see cars with automatic gearbox have high price
# so gear box effect price

# variable model
cars['model'].value_counts()
sns.countplot(x='model',data=cars)
pd.crosstab(cars['model'],columns='counts',normalize=True)
sns.boxplot(x='model',y='price',data=cars)
# cars are distributed over many models
# so considered in modelling

# variable Kilometer
cars['kilometer'].value_counts().sort_index()
sns.countplot(x='kilometer',data=cars)
pd.crosstab(cars['kilometer'],columns='counts',normalize=True)
sns.boxplot(x='kilometer',y='price',data=cars)
sns.distplot(cars['kilometer'],bins=8, kde=False)
# considered in modelling

# variable fuelType
cars['fuelType'].value_counts()
sns.countplot(x='fuelType',data=cars)
pd.crosstab(cars['fuelType'],columns='counts',normalize=True)
sns.boxplot(x='fuelType',y='price',data=cars)
# fuelType effect price

# Variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='counts',normalize=True)
sns.boxplot(x='brand',y='price',data=cars)
# cars are distributed over many brands
# considered for modelling

#variable notRepairedDamage
# yes= car is damaged but not rectified
# no- car is damaged but rectified
cars['notRepairedDamage'].value_counts()
sns.countplot(x='notRepairedDamage',data=cars)
pd.crosstab(cars['notRepairedDamage'],columns='counts',normalize=True)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)
# as expected, the cars that require the damages to be repaired
# falls under low price range

# Removing insgnificant variable
col=['seller','offerType','abtest']
cars=cars.drop(columns=col, axis=1)
cars_copy=cars.copy()

# correlation
cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)    
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


# We are going to built a linear regression and random forest on two data set
# 1). Data obtained by omitting rows with and missing value
# 2) Data obtained by imputing the missing value


# omitting missing value
cars_omit=cars.dropna(axis=0)
# convert categorical variable to dummy variable
cars_omit=pd.get_dummies(cars_omit,drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Model with omited data
x1=cars_omit.drop(['price'], axis='columns', inplace=False)
y1=cars_omit['price']

# ploting the variable price
prices=pd.DataFrame({'1. Before':y1,'2. After': np.log(y1)})
prices.hist()
# histogram of log of Y is more accurate and not diverse so we will use log(y1)
#so
y1=np.log(y1)

# split data into test and train
x_train,x_test,y_train,y_test=train_test_split(x1,y1, test_size=0.3, random_state=3)

# Base line model for omited data
# making a base model by using test data mean vlaue
# This is to set a benchmark and to comparewith our regression plot

# Mean of test data value
base_pred=np.mean(y_test)

# repeating same valuetill length of test data
base_pred= np.repeat(base_pred,len(y_test))

# Finding the RMSE
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test, base_pred))
print(base_root_mean_square_error)

# Linear regression with omited data

#setting intercept as true
lgr=LinearRegression(fit_intercept=True) 

# Model
model_lin1=lgr.fit(x_train,y_train)

#predicting model on test data
cars_prediction_lin1=lgr.predict(x_test)

# computing MSE and RMSE
lin_mse1=mean_squared_error(y_test, cars_prediction_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

# R_squared value
r2_lin_test1=model_lin1.score(x_test,y_test)
r2_lin_train1=model_lin1.score(x_train,y_train)
print(r2_lin_test1,r2_lin_train1)
# difference between train and test is very less that is good thing

# Regression diagnostic- Residual plot analysis
residuals1=y_test-cars_prediction_lin1
sns.regplot(x=cars_prediction_lin1,y=residuals1, scatter=True,fit_reg=False)
# we can most of the values are near 0 which is good
residuals1.describe()
# Mean is near to zero 

# random forest with omitted data
# Model parameters
rf=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=1)

model_rf1=rf.fit(x_train,y_train)

cars_predictions_rf1=rf.predict(x_test)

rf_mse1=mean_squared_error(y_test,cars_predictions_rf1)
rf_rmse1=np.sqrt(rf_mse1)
print(rf_rmse1)

# R squared value
r2_rf_test1=model_rf1.score(x_test,y_test)
r2_rf_train1=model_rf1.score(x_train,y_train)
print(r2_rf_test1,r2_rf_train1)
# This value is better than linear regression model value

# ....................Model building with imputed data.................
cars_imputed=cars.apply(lambda x:x.fillna(x.median()) if x.dtypes=='float' else x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()

cars_imputed=pd.get_dummies(cars_imputed,drop_first=True)

# Model with imputed data
x2=cars_imputed.drop(['price'], axis='columns', inplace=False)
y2=cars_imputed['price']

# ploting the variable price
prices=pd.DataFrame({'1. Before':y2,'2. After': np.log(y2)})
prices.hist()
# histogram of log of Y2 is more accurate and not diverse so we will use log(y1)
#so
y2=np.log(y2)

# split data into test and train
x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y2, test_size=0.3, random_state=3)

# Base line model for imputed data
# making a base model by using test data mean vlaue
# This is to set a benchmark and to compare with our regression plot

# Mean of test data value
base_pred=np.mean(y_test1)

# repeating same valuetill length of test data
base_pred= np.repeat(base_pred,len(y_test1))

# Finding the RMSE
base_root_mean_square_error_imputed=np.sqrt(mean_squared_error(y_test1, base_pred))
print(base_root_mean_square_error_imputed)

# Linear regression with imputed data

#setting intercept as true
lgr2=LinearRegression(fit_intercept=True) 

# Model
model_lin2=lgr2.fit(x_train1,y_train1)

#predicting model on test data
cars_prediction_lin2=lgr2.predict(x_test1)

# computing MSE and RMSE
lin_mse2=mean_squared_error(y_test1, cars_prediction_lin2)
lin_rmse2=np.sqrt(lin_mse2)
print(lin_rmse2)

# R_squared value
r2_lin_test2=model_lin2.score(x_test1,y_test1)
r2_lin_train2=model_lin2.score(x_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)
# difference between train and test is very less that is good thing

# Regression diagnostic- Residual plot analysis
residuals2=y_test1-cars_prediction_lin2
sns.regplot(x=cars_prediction_lin2,y=residuals2, scatter=True,fit_reg=False)
# we can most of the values are near 0 which is good
residuals1.describe()
# Mean is near to zero 

# random forest with imputed data
# Model parameters
rf2=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=1)

model_rf2=rf2.fit(x_train1,y_train1)

cars_predictions_rf2=rf2.predict(x_test1)

rf_mse2=mean_squared_error(y_test1,cars_predictions_rf2)
rf_rmse2=np.sqrt(rf_mse2)
print(rf_rmse2)

# R squared value
r2_rf_test2=model_rf2.score(x_test1,y_test1)
r2_rf_train2=model_rf2.score(x_train1,y_train1)
print(r2_rf_test2,r2_rf_train2)
# This value is better than linear regression model value


################################################################################
# ............................. Final output.................................
print('Metrices for model built from data where missing values were omitted')
print('R squared value for train from linear regression= %s'% r2_lin_train1)
print('R squared value for test from linear regression= %s'% r2_lin_test1)
print('R squared value for train from random forest= %s'% r2_rf_train1)
print('R squared value for test from random forest= %s'% r2_rf_test1)
print('Base RMSE of model built from data where missing values were omitted= %s'%base_root_mean_square_error)
print('RMSE value for test from linear regression =%s'% lin_rmse1)
print('RMSE value for test from Random forest =%s'% rf_rmse1)
print('\n\n')
print('Metrices for model built from data where missing values were imputed')
print('R squared value for train from linear regression= %s'% r2_lin_train2)
print('R squared value for test from linear regression= %s'% r2_lin_test2)
print('R squared value for train from random forest= %s'% r2_rf_train2)
print('R squared value for test from random forest= %s'% r2_rf_test2)
print('Base RMSE of model built from data where missing values were imputed= %s'%base_root_mean_square_error_imputed)
print('RMSE value for test from linear regression =%s'% lin_rmse2)
print('RMSE value for test from Random forest =%s'% rf_rmse2)
