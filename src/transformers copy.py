
import pandas as pd
import json

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.preprocessing import FunctionTransformer, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer






class ClassReplaceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        for column in X.columns:
            if column in self.mapping:
                X_new[column] = X[column].map(self.mapping[column]).fillna(0)
        return X_new




class ColumnMapper_old(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        if isinstance(mapping, dict) and 'mapping_file' in mapping:
            print("`mapping` points to a json file, loading dict...")
            self.mapping = load_dict_from_json(mapping_file_path=mapping['mapping_file'])
            self.mapping = {float(k): int(v) for k, v in self.mapping.items()}
        else:
            self.mapping = mapping
            
        print('`mapping` dict (self.mapping from ColumnMapper): ', self.mapping)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        for column in X_new.columns:
            X_new[column] = X_new[column].map(self.mapping).fillna(0)
        return X_new







class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.columns, axis=1)
        

class JSONMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_file):
        self.mapping_file = mapping_file
        
    def fit(self, X, y=None):
        with open(self.mapping_file, 'r') as f:
            self.mapping = json.load(f)[0]['mapping']
        return self
    
    def transform(self, X):
        # Convert the keys in the mapping to the same type as the first element in the DataFrame
        mapping = {type(X.iloc[0, 0])(k): v for k, v in self.mapping.items()}
        crt = ClassReplaceTransformer(mapping)
        return crt.fit_transform(X)



class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]
    
    
    
class ColumnNamePurger(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Make sure X is a DataFrame
        X = pd.DataFrame(X)

        # Modify the column names
        X.columns = [col.split('__')[-1] for col in X.columns]

        return X
    
    
    
class RegionCodeDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, steps_dict):
        self.mapping_file = steps_dict['feature_engineering']['specific_feat_engineering']\
            ['ordinal_encode_region']['args']['mapping_file']
        print(self.mapping_file)
        
    def fit(self, X, y=None):
        self.mapping = load_dict_from_json(self.mapping_file)
        return self
    
    def transform(self, X):
        print(self.mapping)
        float_mapping = {float(k): int(v) for k, v in self.mapping.items()}
        X['Region_Code'] = X['Region_Code'].map(float_mapping)
        return X


    
class RegionCodeDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, steps_dict):
        self.mapping_file = steps_dict['feature_engineering']['specific_feat_engineering']\
            ['ordinal_encode_region']['args']['mapping_file']
        print(self.mapping_file)
        
    def fit(self, X, y=None):
        self.mapping = load_dict_from_json(self.mapping_file)
        return self
    
    def transform(self, X):
        print(self.mapping)
        float_mapping = {float(k): int(v) for k, v in self.mapping.items()}
        X['Region_Code'] = X['Region_Code'].map(float_mapping)
        return X
    
    
    
class PolicySalesChannelDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, steps_dict):
        self.mapping_file = steps_dict['feature_engineering']['specific_feat_engineering']\
            ['ordinal_encode_policy']['args']['mapping_file']
        print(self.mapping_file)
        
    def fit(self, X, y=None):
        self.mapping = load_dict_from_json(self.mapping_file)
        return self
    
    def transform(self, X):
        print(self.mapping)
        float_mapping = {float(k): int(v) for k, v in self.mapping.items()}
        X['Policy_Sales_Channel'] = X['Policy_Sales_Channel'].map(float_mapping)
        return X
    
    



    
    
    
def load_dict_from_json(mapping_file_path):
    with open(mapping_file_path, 'r') as f:
        return json.load(f)[0]['mapping']



def get_column_transformer_old(columns, transformers, args):
    transformations_list = []
    for transfo, arg in zip(transformers, args):
        transformer_class = globals()[transfo]
        if arg is None:
            transformer_instance = transformer_class()
        else:
            transformer_instance = transformer_class(arg)
        if columns is None:
            transformations_list.append((transformer_instance, 'passthrough'))  # apply to all columns
        else:
            for col in columns:
                transformations_list.append((transformer_instance, [col]))
    return transformations_list



def get_column_transformer_old2(columns, transformers, args):
    transformations_list = []
    for transfo, arg in zip(transformers, args):
        transformer_class = globals()[transfo]
        if arg is None:
            transformer_instance = transformer_class()
        else:
            transformer_instance = transformer_class(arg)
        if columns is None:
            transformations_list.append((transformer_instance, 'passthrough'))  # apply to all columns
        else:
            transformations_list.append((transformer_instance, columns))  # apply to all specified columns
    return transformations_list

def get_column_transformer(columns, transformers, args):
    transformations_list = []
    for column, transfo, arg in zip(columns, transformers, args):
        transformer_class = globals()[transfo]
        if arg is None:
            transformer_instance = transformer_class()
        else:
            transformer_instance = transformer_class(arg)
        transformations_list.append((transformer_instance, [column]))  # apply to specific column
    return transformations_list


def create_transformers_from_dict(steps_dict, key):
    transformers = []
    for step, details in steps_dict[key].items():
        transformer_class = globals()[details['transformer']]
        if 'args' in details and details['args'][0]:  # check if args are provided and not empty
            transformer_instance = transformer_class(**details['args'][0])  # unpack args
        else:
            transformer_instance = transformer_class()
        if 'columns' in details:  # check if columns are provided
            transformers.append((transformer_instance, details['columns']))
        else:  # if no columns provided, apply to all columns
            transformers.append((transformer_instance, 'passthrough'))
    print(transformers)
    return transformers



def get_complete_column_processing_step_old(pipeline_dict, process_round):
    all_transformations = []
    for step, details in pipeline_dict[process_round].items():
        transformers = [details['transformer']] if isinstance(details['transformer'], str) else details['transformer']
        
        if 'columns' in details:
            columns = details['columns']
        else:
            columns = None
            
        column_transformations = get_column_transformer(columns, transformers, details['args'])
        all_transformations.extend(column_transformations)
    return all_transformations



def get_complete_column_processing_step(pipeline_dict, process_round):
    all_transformations = []
    for step, details in pipeline_dict[process_round].items():
        transformers = [details['transformer']] if isinstance(details['transformer'], str) else details['transformer']
        
        if 'columns' in details:
            columns = details['columns']
        else:
            columns = None
            
        # Create a single transformer for all columns
        transformer_class = globals()[transformers[0]]
        if details['args'] is None:
            transformer_instance = transformer_class()
        else:
            transformer_instance = transformer_class(details['args'])
        
        all_transformations.append((transformer_instance, columns))
    return all_transformations


def get_column_selection(pipeline_dict, selection):
    column_selection = pipeline_dict['column_selection'][selection]
    transformer_instance = ColumnSelector(column_selection)
    return [(transformer_instance, column_selection)]




def create_preprocessing_pipeline(steps_dict):
    cleaning_instructions = get_complete_column_processing_step(steps_dict, 'cleaning')
    cleaning_columns_transformer = make_column_transformer(*cleaning_instructions, remainder='drop')
    
    column_passthru_instructions = get_column_selection(steps_dict, 'preprocess_passthrough_1')
    cleaning_columns_passthru = make_column_transformer(*column_passthru_instructions)
    
    combined_features = FeatureUnion([
        ('column_selection', cleaning_columns_transformer),
        ('column_transformation', cleaning_columns_passthru)
    ])

    column_name_purger = ColumnNamePurger()

    preprocessing_pipeline = Pipeline([
        ('combined_features', combined_features),
        ('column_name_purger', column_name_purger)
    ])

    return preprocessing_pipeline


def create_feat_engineering_pipeline(steps_dict):
    cleaning_instructions = get_complete_column_processing_step(steps_dict['feature_engineering'], 'specific_f_e')
    cleaning_columns_transformer = make_column_transformer(*cleaning_instructions, remainder='drop')
    
    column_passthru_instructions = get_column_selection(steps_dict, 'preprocess_passthrough_2')
    cleaning_columns_passthru = make_column_transformer(*column_passthru_instructions)
    
    combined_features = FeatureUnion([
        ('column_selection', cleaning_columns_transformer),
        ('column_transformation', cleaning_columns_passthru)
    ])

    column_name_purger = ColumnNamePurger()

    f_e_pipeline = Pipeline([
        ('combined_features', combined_features),
        ('column_name_purger', column_name_purger)
    ])

    return f_e_pipeline