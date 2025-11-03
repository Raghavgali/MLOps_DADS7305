import pytest
import torch 
import numpy as np
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset
from src.train_and_save_model import download_data, preprocess_data, data_loader
from src.train_and_save_model import train_model, evaluate_model
from src.train_and_save_model import update_model_version, get_model_version
from src.train_and_save_model import ensure_folder_exists, save_model_to_gcs
from google.cloud import storage 


# ----------------- Test Download ----------------- #
# Test the download_data function to ensure it correctly downloads and returns data
def test_download_data():
    X, y = download_data()

    # Check if the data is downloaded correctly and matches expected formats
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.size > 0
    assert y.size > 0
    assert X.shape[0] == y.shape[0]

# ----------------- Test Preprocess ----------------- #
# Test the preprocess_data function to ensure it correctly preprocesses the data
def test_preprocess_data():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Assert that the preprocessing splits the data correctly
    assert X_train.shape == (X_train.shape[0], 1, 8, 8)
    assert X_test.shape == (X_test.shape[0], 1, 8, 8)

    assert X_train.shape[0] + X_test.shape[0] == X.shape[0] # Rows in train and test should total original rows
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0] # Rows in train and test labels should total original labels


# ----------------- Test DataLoader ----------------- #
# Test the data_loader function to ensure it correctly loads the data into Data Loader
def test_data_loader():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    train_dataloader, test_data_loader = data_loader(X_train, X_test, y_train, y_test)

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(test_data_loader, DataLoader)

    train_batch = next(iter(train_dataloader))
    test_batch = next(iter(test_data_loader))

    assert len(train_batch) == 2
    assert len(test_batch) == 2


# ----------------- Test Train model ----------------- #
# Test the train_model function to ensure it correctly trains the model
def test_train_model():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    train_dataloader, _ = data_loader(X_train, X_test, y_train, y_test)

    # Train the model 
    model = train_model(train_dataloader, num_epochs=1)

    assert isinstance(model, torch.nn.Module)
    sample_input = X_train[0].unsqueeze(0).to(next(model.parameters()).device)
    output = model(sample_input)
    assert output.shape[-1] == 10 # Output should match number of classes


# ----------------- Evaluate model ----------------- #
# Test the evaluate_model function to ensure it correctly evaluates the model
def test_evaluate_model():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    train_dataloader, test_data_loader = data_loader(X_train, X_test, y_train, y_test)

    # Train the model 
    model = train_model(train_dataloader, num_epochs=1)
    model = model.eval()

    assert model.training == False, "Model should be in eval mode during evaluation"
    # evaluate model 
    accuracy = evaluate_model(model, test_data_loader)

    # Assertions
    assert isinstance(accuracy, float), "Accuracy should be a float value"
    assert 0.0 <= accuracy <= 1.0, "Accuracy must be between 0 and 1"


# ----------------- Test Model versioning ----------------- #
# mock test a practice exam designed to mimic the format and difficulty level of a real exam
# This function tests the get_model_version function responsible for retrieving the version of the model stored in Google Cloud Storage.
def test_get_model_version():
    # Patch the GCP storage client to prevent actual network operations during the test.
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()  # Create a mock bucket object.
        mock_blob = MagicMock()    # Create a mock blob object to represent the file in the storage.

        # Configure mock objects to return other mocks when methods are called.
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Set the test inputs for actual function calls
        bucket_name = "bucket-test"
        version_file_name = "version.txt"

        # Simulate the scenario where the version file exists in the storage
        mock_blob.exists.return_value = True
        version = get_model_version(bucket_name, version_file_name)
        mock_blob.download_as_text.return_value = '1'  # Simulate blob returning version '1' as text

        # Check if the correct version is retrieved and the corresponding methods are called on the mock
        assert version == 1
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.download_as_text.assert_called_once()
        
        # Reset mocks to clear call history before the next test
        mock_storage_client.reset_mock()
        mock_bucket.reset_mock()
        mock_blob.reset_mock()
        
        # Test scenario where the version file does not exist
        mock_blob.exists.return_value = False
        version = get_model_version(bucket_name, version_file_name)
        mock_blob.download_as_text.return_value = '0'  # No file to download, should return 0

        # Ensure it handles the absence of the version file correctly
        assert version == 0
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.download_as_text.assert_not_called()

# ----------------- Test Update Model version ----------------- #
# This function tests the update_model_version function that updates the version of the model stored in Google Cloud Storage.
def test_update_model_version():
    # Patch the GCP storage client to prevent actual network operations during the test.
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()  # Create a mock bucket object.
        mock_blob = MagicMock()    # Create a mock blob object to represent the file in the storage.

        # Configure mock objects to return other mocks when methods are called.
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        bucket_name = 'bucket-test'
        version_file_name = 'version.txt'
        new_version = 2
        
        # Test successful update of the model version
        result = update_model_version(bucket_name, version_file_name, new_version)
        # Assert the function returns true indicating success
        assert result == True
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.upload_from_string.assert_called_once_with(str(new_version))
        
        # Reset mocks to clear call history for further tests
        mock_storage_client.reset_mock()
        mock_bucket.reset_mock()
        mock_blob.reset_mock()
        
        # Test error handling with an invalid version (not an integer)
        with pytest.raises(ValueError):
            update_model_version(bucket_name, version_file_name, 'invalid_version')
        
        # Simulate an exception during the blob upload to test error handling
        mock_blob.upload_from_string.side_effect = Exception("Upload failed")
        result = update_model_version(bucket_name, version_file_name, new_version)
        # Assert the function returns false indicating failure
        assert result == False
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.upload_from_string.assert_called_once_with(str(new_version))


# ----------------- Test Ensure Folder Exists ----------------- #
# Test ensure_folder_exists function to verify it correctly ensures the presence of a folder in the storage
def test_ensure_folder_exists():
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        folder_name = "trained_models"
        
        # When folder does not exist
        mock_blob.exists.return_value = False
        ensure_folder_exists(mock_bucket, folder_name)
        mock_bucket.blob.assert_called_with(f"{folder_name}/")
        mock_blob.upload_from_string.assert_called_once_with('')
        
        # Reset the mock for the next test
        mock_blob.reset_mock()
        
        # When folder exists
        mock_blob.exists.return_value = True
        ensure_folder_exists(mock_bucket, folder_name)
        mock_bucket.blob.assert_called_with(f"{folder_name}/")
        mock_blob.upload_from_string.assert_not_called()

# ----------------- Test Save model to GCS ----------------- #
# This function tests the 'save_model_to_gcs' function to ensure it correctly saves a model to Google Cloud Storage.
def test_save_model_to_gcs():
    # Prepare and train a lightweight torch model
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    train_dataloader, _ = data_loader(X_train, X_test, y_train, y_test)
    model = train_model(train_dataloader, num_epochs=1)

    
    # 'patch' is used to temporarily replace the 'google.cloud.storage.Client' class with a mock, so that no real network operations are performed.
    with patch('google.cloud.storage.Client') as mock_storage_client:
        # Create a mock bucket object. This mock simulates the bucket where the model will be stored in GCS.
        mock_bucket = MagicMock()
        # Create a mock blob object. This simulates the file or object within the GCS bucket.
        mock_blob = MagicMock()
        
        # Set up the return values for when the storage client, bucket, and blob are called.
        # This ensures that when the save_model_to_gcs function tries to interact with GCS, it uses these mock objects instead.
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Configure the mock blob to simulate a scenario where the blob does not already exist in the GCS bucket.
        mock_blob.exists.return_value = False
        
        # Call the function 'save_model_to_gcs' with the mock model and specific bucket and blob names.
        # This is the function you are testing, which should perform the actual saving of the model.
        save_model_to_gcs(model, 'bucket-test', 'blob-test')
        
        # Ensure the storage client was initialized exactly once.
        mock_storage_client.assert_called_once()
        # Check that the bucket was fetched exactly once with the specified name.
        mock_storage_client.return_value.bucket.assert_called_once_with('bucket-test')
        
        # Assert that the blob method was called exactly twice (once for checking existence, once for uploading).
        # This line checks that the blob method is called with the correct parameters at least once.
        assert mock_bucket.blob.call_count == 2
        # These calls ensure that both blob invocations were with the expected names.
        mock_bucket.blob.assert_any_call('trained_models/')
        mock_bucket.blob.assert_any_call('blob-test')
        # Verify that the blob's 'upload_from_filename' method was called once with the filename 'model.joblib' to upload the model.
        mock_blob.upload_from_filename.assert_called_once_with('model.joblib')