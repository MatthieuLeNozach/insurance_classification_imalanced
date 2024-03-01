
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




BINARY_COLUMNS = ['Driving_License', 'Previously_Insured',]










def process_column_transformer(df):
    binary_columns = BINARY_COLUMNS
    
    binaries_pipeline = make_pipeline(
        IterativeImputer(),
    )

    gender_pipeline = make_pipeline(
        OrdinalEncoder(categories=[['Male', 'Female']]),
        IterativeImputer(),
    )
        
    vehicle_damage_pipeline = make_pipeline(
        OrdinalEncoder(categories=[['No', 'Yes']]),
        IterativeImputer(),
    )

    vehicle_age_pipeline = make_pipeline(
        OrdinalEncoder(categories=[sorted(df['Vehicle_Age'].unique())]),
        IterativeImputer(),

    )

    region_code_pipeline = make_pipeline(
        IterativeImputer(),
        EqualFrequencyDiscretiser(q=4)
    )

    policy_sales_channel_pipeline = make_pipeline(
        IterativeImputer(),
        EqualFrequencyDiscretiser(q=4)
    )

    continuous_pipeline = make_pipeline(
        IterativeImputer(),
        RobustScaler(),
    )

    age_pipeline = make_pipeline(
        IterativeImputer(),
        StandardScaler(),
    )

    column_transformer = make_column_transformer(
        (binaries_pipeline, binary_columns),
        (gender_pipeline, ['Gender']),
        (vehicle_damage_pipeline, ['Vehicle_Damage']),
        (vehicle_age_pipeline, ['Vehicle_Age']),
        (region_code_pipeline, ['Region_Code']),
        (policy_sales_channel_pipeline, ['Policy_Sales_Channel']),
        (continuous_pipeline, ['Annual_Premium']),
        (age_pipeline, ['Age']),
    )

    
    return column_transformer





def generate_balanced_pipeline(train):
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
        EqualFrequencyDiscretiser(q=4)
    )

    policy_sales_channel_pipeline = make_pipeline(
        EqualFrequencyDiscretiser(q=4)
    )

    continuous_pipeline = make_pipeline(
        RobustScaler(),
    )

    age_pipeline = make_pipeline(
        StandardScaler(),
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
        ('passthrough', ['Driving_License', 'Previously_Insured',]),
    )

    feature_engineering_pipeline = make_pipeline(column_transformer, 
                                                 iterative_imputer,
                                                 VarianceThreshold(threshold=0.1)
    )

    best_knn = KNeighborsClassifier(n_neighbors=7, weights='uniform', p=2)
    best_gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1)
    best_sgd = SGDClassifier(alpha=0.001, learning_rate='optimal', loss='modified_huber', max_iter=1000, penalty='elasticnet')

    voting_clf = VotingClassifier(
        estimators=[('GBC', best_gb), ('KNN', best_knn), ('SGD', best_sgd)],
        voting='soft',
        n_jobs=-1,
        verbose=3
    )

    vc_clf_pipeline = Pipeline(steps=[
        ('preprocessor', feature_engineering_pipeline),
        ('classifier', voting_clf)
    ])

    imb_fe_pipeline = imbPipeline([
        ('column_transformer', column_transformer),
        ('variance_threshold', VarianceThreshold(threshold=0.09))
    ])

    return vc_clf_pipeline, imb_fe_pipeline