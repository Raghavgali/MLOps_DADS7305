import os 
import pickle 
import torch 
import pandas as pd 
from torch import nn 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# For local testing 
# WORKING_DIR = "./working_data"
# MODEL_DIR = "./model"

WORKING_DIR = "/opt/airflow/working_data"
MODEL_DIR = "/opt/airflow/model"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data() -> str:
    """
    Load CSV and persist raw dataframe to a pickle file.
    Returns path to saved file.
    """
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "breast_cancer.csv",
    )

    df = pd.read_csv(csv_path)

    out_path = os.path.join(WORKING_DIR, "raw.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    return out_path


def data_preprocessing(file_path: str) -> str:
    """Load Dataframe, split and scale and save(X_train, X_test, y_train, y_test) to pickle.
    Returns path to saved file.
    """
    with open(file_path, "rb") as f:
        df = pickle.load(f)

    X = df.drop('target', axis=1)
    y = df['target']

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    out_path = os.path.join(WORKING_DIR, "preprocessed.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    return out_path


def separate_data_outputs(file_path: str) -> str:
    """Passthrough: kept so the DAG composes cleanly.
    """
    return file_path


def build_model(file_path: str, filename: str) -> str:
    """
    Train a neural network and save to MODEL_DIR/filename. Returns model path
    """
    
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(30, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNetwork().to(device)

    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(5):
        model.train()
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_logits = model(X_train)
        y_pred = torch.round(y_logits)
        loss = loss_fn(y_logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model_path = os.path.join(MODEL_DIR, filename)
    torch.save(model.state_dict(), model_path)
    
    return model_path 


def load_model(file_path: str, filename: str) -> int:
    """
    Load saved model and test set, print score and return the first prediction as int.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(30, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model_name = os.path.join(MODEL_DIR, filename)
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_name))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    first_pred = None

    with torch.inference_mode():
        X_test, y_test = X_test.to(device), y_test.to(device)
        output_pred = model(X_test)
        predicted = torch.round(output_pred)
        if first_pred is None:
            first_pred = int(predicted[0].item())
        correct += (predicted == y_test).sum().item()
        total += y_test.size(0)

    accuracy = correct / total
    print(f"Model Accuracy on test set: {accuracy:.4f}")

    return first_pred



# if __name__ == "__main__":
#     try:
#         data_path = load_data()
#         print("Data loader works")
#         data_preprocessing(file_path=data_path)
#         print("Data Preprocessing works")
#         build_model(file_path="/Users/raghavg/Desktop/MLOps/Lab2_assignment/dags/working_data/preprocessed.pkl", filename="test_model")
#         print("Build Model Works")
#         load_model(file_path="/Users/raghavg/Desktop/MLOps/Lab2_assignment/dags/working_data/preprocessed.pkl", filename="test_model")
#         print("Load Model works")
#     except Exception as e:
#         print(f"Exception occurred: {e}")