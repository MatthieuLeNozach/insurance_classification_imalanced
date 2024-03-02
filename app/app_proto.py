import sys
sys.path.insert(0, '../src')

from sklearn import set_config
set_config(display='diagram')

from sklearn import set_config
set_config(transform_output='pandas')


import pickle
import numpy as np
import pandas as pd

from joblib import load
from IPython.display import display


processed_pipeline = load('../models/user_input_preprocessing_pipeline.joblib')
model = load('../models/vc_knn_gbc_sgd_standalone_balanced_trained.joblib')

user_input_raw = {
    "Policy_Sales_Channel": "26.0",
    "Region_Code": "28.0",
    "Age": "44",
    "Previously_Insured": "1",
    "Vehicle_Age": "< 1 Year",
    "Vehicle_Damage": "No", 
    "Annual_Premium": "40454.0",
    "Gender": "Female",
}



pipeline_columns = [
    'pipeline-1__Gender', 'pipeline-2__Vehicle_Damage', 
    'pipeline-3__Vehicle_Age', 'pipeline-4__Region_Code', 
    'pipeline-5__Policy_Sales_Channel', 'pipeline-6__Annual_Premium', 
    'pipeline-7__Age', 'passthrough__Previously_Insured'
]




user_input_raw['Policy_Sales_Channel'] = float(user_input_raw['Policy_Sales_Channel'])
user_input_raw['Region_Code'] = float(user_input_raw['Region_Code'])
user_input_raw['Age'] = int(user_input_raw['Age'])
user_input_raw['Previously_Insured'] = int(user_input_raw['Previously_Insured'])
user_input_raw['Annual_Premium'] = float(user_input_raw['Annual_Premium'])

user_input_df = pd.DataFrame(user_input_raw, index=[0])
preprocessed_user_input = processed_pipeline.transform(user_input_df)
display(preprocessed_user_input)


prediction = model.predict(preprocessed_user_input)
print(prediction)



model_columns =[
    'Policy_Sales_Channel',
    'Region_Code',
    'Age',
    'Previously_Insured',
    'Vehicle_Age',
    'Vehicle_Damage',
    'Annual_Premium',
    'Gender'
]