import kaggle

# Download the dataset from Kaggle and unzip it
kaggle.api.authenticate()
kaggle.api.dataset_download_files('arashnic/imbalanced-data-practice', path='../data/', unzip=True)