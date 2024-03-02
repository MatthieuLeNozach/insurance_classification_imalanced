import unittest
import requests
from joblib import load
import pandas as pd
import numpy as np


class TestModelPredictions(unittest.TestCase):
    def test_model_discrepancies(self):
        loaded_voting_clf = load('../models/vc_knn_gbc_sgd_standalone_balanced_trained.joblib')
        loaded_preprocessing_pipeline = load('../models/user_input_preprocessing_pipeline.joblib')
        
        train = pd.read_csv('../data/aug_train.csv')
        positive_responses = train[train['Response'] == 1]
        data_subset = positive_responses.sample(100)
        
        X = data_subset.drop(columns=['Response'])
        X_preprocessed = loaded_preprocessing_pipeline.transform(X)
        y_pred_loaded_model = loaded_voting_clf.predict(X_preprocessed)
        
        endpoint = 'http://localhost:3000/api/predict'
        requests_data = [row.drop('Response', errors='ignore').to_dict() for _, row in data_subset.iterrows()]
        responses = [requests.post(endpoint, json=data, headers={'Content-Type': 'application/json'}).json() for data in requests_data]
        y_pred_distant = [response['prediction'] for response in responses]
        
        comparison_df = pd.DataFrame({
            'TrueLabel': data_subset['Response'],
            'LoadedModelPrediction': y_pred_loaded_model,
            'DistantModelPrediction': y_pred_distant
            }, index=data_subset.index
        )
        
        same_model_discrepancies = comparison_df.loc[comparison_df['LoadedModelPrediction'] != comparison_df['DistantModelPrediction']]
        self.assertLessEqual(same_model_discrepancies.shape[0], 2, "Discrepancies between the notebook model and the Flask model predictions exceed the limit")
        
        y_true_vs_y_pred_distant = comparison_df.loc[comparison_df['TrueLabel'] != comparison_df['DistantModelPrediction']]     
        print(f"Warning: Discrepancies between the true label and the Flask model: {y_true_vs_y_pred_distant.shape[0]}/100")

if __name__ == '__main__':
    unittest.main()