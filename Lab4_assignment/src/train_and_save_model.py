import os
import joblib
import torch 
import torch.nn as nn 
import torch.optim as optim
from dotenv import load_dotenv
from google.cloud import storage
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Load environment variables from a .env file that should be located in the same directory as this script
load_dotenv()

# Retrieve the values of environment variables defined in the .env file
BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')  # Google Cloud Storage bucket name
VERSION_FILE_NAME = os.getenv('VERSION_FILE_NAME')  # Name of the file where the model version is stored


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def download_data():
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    return X, y


def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_train = X_train.view(-1, 1, 8, 8)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    X_test = X_test.view(-1, 1, 8, 8)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test


def data_loader(X_train, X_test, y_train, y_test):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    return train_dataloader, test_dataloader


class TrainMNIST(nn.Module):
    def __init__(self):
        super(TrainMNIST, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

def train_model(train_dataloader, num_epochs=10, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrainMNIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train() 
        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model


def evaluate_model(model, test_dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100: .2f}%")
    return accuracy


# Function to retrieve the current model version from Google Cloud Storage
def get_model_version(bucket_name, version_file_name):
    storage_client = storage.Client() # Create a GCP Storage Client

    bucket = storage_client.bucket(bucket_name) # Access the specified bucket

    blob = bucket.blob(version_file_name) # Access the specified blob (binary large objects) within the bucket

    if blob.exists():
        version_as_string = blob.download_as_text() # Retrieve the version number as text
        version = int(version_as_string) # Convert the version number to an integer
    else:
        version = 0 # If the blob does not exist, set version to 0
    return version


# Function to update the model version in Google Cloud Storage
def update_model_version(bucket_name, version_file_name, version):
    if not isinstance(version, int):
        raise ValueError("Version must be an integer") # Ensure that the version is an integer
    try:
        storage_client = storage.Client() # Create a GCP Storage client 
        bucket = storage_client.bucket(bucket_name) # Access the specified bucket
        blob = bucket.blob(version_file_name) # Access the specified blob (file) within the bucket 
        blob.upload_from_string(str(version)) # Upload the new version number as a string
        return True
    except Exception as e:
        print(f"Error updating model version: {e}") # Print any error that occurs
        return False
    

# Function to ensure that a specific folder exists within a bucket in Google Cloud Storage
def ensure_folder_exists(bucket, folder_name):
    blob = bucket.blob(f"{folder_name}/") # Define the blob path as a folder
    if not blob.exists():
        blob.upload_from_string('') # If the folder does not exist, create it by updating an empty string
        print(f"Created folder: {folder_name}")


# Function to save the trained model both locally and to Google Cloud Storage
def save_model_to_gcs(model, bucket_name, blob_name):
    joblib.dump(model, "model.joblib") # Save the model locally using joblib

    # Initialize storage client and access the specified bucket 
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    ensure_folder_exists(bucket, "trained_models") # Ensure the "trained_models" folder exists in the bucket

    blob = bucket.blob(blob_name) # Create a blob for the model in the specified path
    blob.upload_from_filename('model.joblib') # Upload the locally saved model to Google Cloud Storage

def main():
    # Retrieve and update model version
    current_version = get_model_version(BUCKET_NAME, VERSION_FILE_NAME)
    new_version = current_version + 1

    # Download Data 
    X, y = download_data()

    # Preprocess Data 
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Dataloader 
    train_dataloader, test_dataloader = data_loader(X_train, X_test, y_train, y_test)

    # Training
    model = train_model(train_dataloader=train_dataloader)

    # Evaluation
    evaluate_model(model, test_dataloader=test_dataloader)

    # Save the model with a new version and timestamp in GCS
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    blob_name = f"trained_models/model_v{new_version}_{timestamp}.joblib"
    save_model_to_gcs(model, BUCKET_NAME, blob_name)
    print(f"Model saved to gs://{BUCKET_NAME}/{blob_name}")

    # Update the model version in GCS if saving was successful
    if update_model_version(BUCKET_NAME, VERSION_FILE_NAME, new_version):
        print(f"Model version updated to {new_version}")
        print(f"MODEL_VERSION_OUTPUT: {new_version}")
    else:
        print("Failed to update model version")


if __name__ == "__main__":
    main()
