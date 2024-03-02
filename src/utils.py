import pandas as pd
import json
import warnings
import logging



def make_feature_bin_mapping_from_impact_on_target(df, feature, target='Response', bins=2):
    """
    Creates a mapping from unique feature values to discrete bins based on their impact on the target variable.
    It will try to cut into `bins` bins, but may return fewer if there are not enough unique values.
    """
    contingency_table = pd.crosstab(df[feature], df[target], normalize='index')[1]
    bins = pd.qcut(contingency_table, bins, labels=False, duplicates='drop')
    return bins.to_dict()  # Convert the Series to a dictionary


def make_feature_mappings_file(df):
    """
    """
    policy_sales_channel_mapping = make_feature_bin_mapping_from_impact_on_target(df, 'Policy_Sales_Channel')
    region_code_mapping = make_feature_bin_mapping_from_impact_on_target(df, 'Region_Code', bins=3)
    
    policy_sales_channel_mapping = [
        {'name': 'Policy_Sales_Channel', 'mapping': policy_sales_channel_mapping},
    ]
    region_code_mapping = [
        {'name': 'Region_Code', 'mapping': region_code_mapping}
    ]
    
    with open('policy_sales_channel_mapping.json', 'w') as f:
        json.dump(policy_sales_channel_mapping, f)

    with open('region_code_mapping.json', 'w') as f:
        json.dump(policy_sales_channel_mapping, f)
        
        
        
def load_dict_from_json(mapping_file_path):
    with open(mapping_file_path, 'r') as f:
        return json.load(f)

    
    
def is_fitted(estimator):
    try:
        # Check if the estimators_ attribute exists
        getattr(estimator, "estimators_")
    except AttributeError:
        # The model is not fitted
        return False
    return True


def warn_with_log(message, category, filename, lineno, file=None, line=None):
    log = logging.getLogger(filename)
    log.warning(f'{message} at line {lineno}')