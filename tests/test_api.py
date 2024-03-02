import unittest
import requests
import json

class TestFlaskApi(unittest.TestCase):
    def test_health_check(self):
        response = requests.get('http://localhost:3000/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, 'The server is up and running')

    def test_predict_form(self):
        data = {
            'Gender': 'Male',
            'Vehicle_Damage': 'Yes',
            'Vehicle_Age': '< 1 Year',
            'Region_Code': 28.0,
            'Policy_Sales_Channel': 152.0,
            'Annual_Premium': 30000.0,
            'Age': 25,
            'Previously_Insured': 0
        }
        response = requests.post('http://localhost:3000/predict', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('Eligible Customer', response.text)

    def test_api_predict(self):
        data = {
            'Gender': 'Male',
            'Vehicle_Damage': 'Yes',
            'Vehicle_Age': '< 1 Year',
            'Region_Code': 28.0,
            'Policy_Sales_Channel': 152.0,
            'Annual_Premium': 30000.0,
            'Age': 25,
            'Previously_Insured': 0
        }
        response = requests.post('http://localhost:3000/api/predict', data=json.dumps(data), headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json()['prediction'], int)

if __name__ == '__main__':
    unittest.main()