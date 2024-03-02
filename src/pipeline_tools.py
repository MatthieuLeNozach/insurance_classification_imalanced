
import pandas as pd
import json

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.compose import make_column_transformer

from sklearn.preprocessing import FunctionTransformer, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from feature_engine.discretisation import EqualFrequencyDiscretiser

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier 

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.pipeline import Pipeline as imbPipeline
from feature_engine.discretisation import EqualFrequencyDiscretiser

from transformers import IntToFloatTransformer, ColumnNamePurger

import sys
sys.path.insert(0, '../src')

from sklearn import set_config
set_config(display='diagram')

from sklearn import set_config
set_config(transform_output='pandas')


BINARY_COLUMNS = ['Driving_License', 'Previously_Insured',]  
    
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

def generate_user_input_pipeline(train):
    gender_pipeline = make_pipeline(
        OrdinalEncoder(categories=[['Male', 'Female']]),
    )
        
    vehicle_damage_pipeline = make_pipeline(
        OrdinalEncoder(categories=[['No', 'Yes']]),
    )

    vehicle_age_pipeline = make_pipeline(
        OrdinalEncoder(categories=[sorted(train['Vehicle_Age'].unique())]),
    )

    region_code_pipeline = make_pipeline(
        EqualFrequencyDiscretiser(q=4),
    )

    policy_sales_channel_pipeline = make_pipeline(
        EqualFrequencyDiscretiser(q=4),
    )

    continuous_pipeline = make_pipeline(
        RobustScaler(),
    )

    age_pipeline = make_pipeline(
        StandardScaler(),
    )

    binaries_pipeline = make_pipeline(
    )

    iterative_imputer = IterativeImputer()

    column_transformer = make_column_transformer(
        (gender_pipeline, ['Gender']),
        (vehicle_damage_pipeline, ['Vehicle_Damage']),
        (vehicle_age_pipeline, ['Vehicle_Age']),
        (region_code_pipeline, ['Region_Code']),
        (policy_sales_channel_pipeline, ['Policy_Sales_Channel']),
        (continuous_pipeline, ['Annual_Premium']),
        (age_pipeline, ['Age']),
        ('passthrough', ['Previously_Insured']),
        #(binaries_pipeline, ['Previously_Insured',]),
        
    )

    feature_engineering_pipeline = make_pipeline(column_transformer, 
                                                iterative_imputer,
                                                VarianceThreshold(threshold=0.09),
                                                #ColumnNamePurger()
    )

    return feature_engineering_pipeline

