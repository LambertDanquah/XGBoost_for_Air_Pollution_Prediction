# MY FIRST PROJECT IN ADVANCED ARTIFICIAL INTELLIGENCE (AAI) COURSE
# Urban Air Pollution Prediction    

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# Extract and read urban pollution data from path

train_path = 'C:/Users/Lambert/Desktop/AAI_project_1/Train.csv'
training_data = pd.read_csv(train_path)

#fill in blank spaces in the training dataset
training_data = training_data.fillna(training_data.mean())

#set target for the model
y = training_data.target

training_data['Date'] = pd.to_datetime(training_data['Date']).values.astype(np.int64)

#extract input features of the dataset and drop less relevant columns
column_features = ['Place_ID X Date', 'Place_ID', 'target', 'target_min', 'target_max',
 'target_variance', 'target_count', 'L3_CH4_CH4_column_volume_mixing_ratio_dry_air', 'L3_CH4_aerosol_height', 
'L3_CH4_aerosol_optical_depth', 'L3_CH4_sensor_azimuth_angle', 'L3_CH4_solar_zenith_angle',
'L3_CH4_sensor_zenith_angle', 'L3_CH4_solar_azimuth_angle']


#, 'L3_HCHO_sensor_zenith_angle',
  #                'L3_AER_AI_sensor_azimuth_angle', 'L3_SO2_absorbing_aerosol_index'

training_data = training_data.drop(column_features, axis=1)
x = training_data

#using an ML model

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=0, test_size = 0.33)


xgb_model = XGBRegressor(seed = 27, subsample = 0.6, n_estimators = 600, max_depth = 20, learning_rate = 0.04,
                         colsample_bytree = 0.8, min_child_weight = 10, reg_lambda = 4, gamma = 4)


xgb_model.fit(train_x, train_y) 

#prediction_test_x = xgb_model.predict(test_x)


#MSE = mean_squared_error(test_y, prediction_test_x)
#RMSE = math.sqrt(MSE)
#print('Root Mean Square error = ' + str(RMSE) )


#loading dataset to be predicted
dataset_path = 'C:/Users/Lambert/Desktop/AAI_project_1/Test.csv'
actual_dataset = pd.read_csv(dataset_path)
actual_dataset = actual_dataset.fillna(actual_dataset.mean())
actual_dataset['Date'] = pd.to_datetime(actual_dataset['Date']).values.astype(np.int64)

actual_column_features = ['Place_ID X Date', 'Place_ID', 'L3_CH4_CH4_column_volume_mixing_ratio_dry_air', 'L3_CH4_aerosol_height', 
'L3_CH4_aerosol_optical_depth', 'L3_CH4_sensor_azimuth_angle', 'L3_CH4_solar_zenith_angle',
'L3_CH4_sensor_zenith_angle', 'L3_CH4_solar_azimuth_angle']

actual_dataset = actual_dataset.drop(actual_column_features, axis = 1)

                                                
model_predictions = xgb_model.predict(actual_dataset)

pollution_prediction = 'C:/Users/Lambert/Desktop/AAI_project_1/Real_Last_Prediction.csv'
my_prediction = pd.DataFrame(model_predictions)
my_prediction.to_csv(pollution_prediction)


print('Done!')

#  The Extreme Gradient Boosting (XG Boost) prediction ends here!
#  The comments below are the hyperparameter variations I used in order to increase my model's accuracy
#  I started with RandomForest and later changed to XG Boost   






'''                      
# eta = 0.2, #early_stopping_rounds = 60

#parameters = {'n_estimators' : [ 50, 500, 10], 'max_depth' : [ 7, 20, 2], 'learning_rate' : [0.05, 0.4],
#             'colsample_bytree' : [0.5, 1], 'subsample' : [0.6, 1], 'min_child_weight': [1, 10], 'eta' : [3, 10]}
 
#Create the Parameter Grid
#parameters = {'n_estimators' : [50, 500, 10],
 #             'max_depth' : [5, 20, 5], 'learning_rate' : [0.05, 0.3, 0.05]}



#xgb_model = XGBRegressor(seed = 27)
#my_grid_search = GridSearchCV(estimator = xgb_model, param_grid = parameters, verbose = 2, cv = 4)  
#my_grid_search.fit(train_x, train_y)
#print(my_grid_search.best_params_)


#{'colsample_bytree': 1, 'learning_rate': 0.05, 'max_depth': 20, 'n_estimators': 500, 'subsample': 0.6}
#xgb_model = XGBRegressor(seed = 27, subsample = 1, n_estimators = 500, min_child_weight = 1, max_depth = 20,
#                         learning_rate = 0.05, eta = 3, colsample_bytree = 0.50)



'L3_HCHO_sensor_zenith_angle' = Root Mean Square error = 28.307168800363275
'L3_HCHO_sensor_zenith_angle' and 'L3_AER_AI_sensor_azimuth_angle', and 'L3_SO2_absorbing_aerosol_index'
(with lambda and gamma) = Root Mean Square error = 28.08146007299745


'L3_HCHO_sensor_zenith_angle', 'L3_AER_AI_sensor_azimuth_angle',
'L3_SO2_absorbing_aerosol_index', 'L3_CLOUD_sensor_zenith_angle', Root Mean Square error = 28.05651213807113

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=0, test_size = 0.3)
Root Mean Square error = 27.752205570780678


#This gives a RMSE = 28.477361617158092 (an improvement from 29.130168192464595)
{'subsample': 0.6,'n_estimators': 500, 'max_depth': 20, 'learning_rate': 0.05, 'eta': 3, 'colsample_bytree': 1,
               'min_child_weight': 10,   }
               

This givesRoot Mean Square error = 28.325367857311875                          
subsample = 0.6, n_estimators = 500, max_depth = 20, learning_rate = 0.05,
                         eta = 3, colsample_bytree = 1, min_child_weight = 10, gamma = 4)



Root Mean Square error = 27.541356253335675
xgb_model = XGBRegressor(seed = 27, subsample = 0.6, n_estimators = 600, max_depth = 20, learning_rate = 0.04,
                         colsample_bytree = 0.8, min_child_weight = 10, reg_lambda = 4, gamma = 4)

test_size = 0.33 Root Mean Square error = 27.43641495838591
'''
