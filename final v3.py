#----------------------------------------------------------------------
import torch
import torch.nn as nn
import math
import pandas as pd
from datetime import datetime as dt, timedelta
from torch.utils.data import Dataset, DataLoader
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
seed = 42  # choose any seed you prefer
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#----------------------------------------------------------------------
# hyperparameters for dataset and dataloader
input_size = 20
target_size = 2
batch_size = 16
dropout_rate = 0.2

# hyperparameters for training, testing
learning_rate = 0.01
num_epochs = 5

#----------------------------------------------------------------------
class PriceDataset(Dataset):
    def __init__(self, item, timespan, start_date_str, end_date_str, input_window_size, target_window_size):
        self.directory = f'C:/Github/DL-FinalProject/csvfiles/{item}/'
        self.input_window_size = input_window_size
        self.target_window_size = target_window_size
        self.columns = [1, 4]  # Adjust as needed for zero-based indexing in NumPy
        self.data = self.load_data(start_date_str, end_date_str)

    def load_data(self, start_date_str, end_date_str):
        start_date = dt.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = dt.strptime(end_date_str, '%Y-%m-%d').date()
        all_data = []

        for filename in os.listdir(self.directory):
            # Extract date from filename
            file_date_str = '-'.join(filename.split('-')[2:]).split('.')[0]
            file_date = dt.strptime(file_date_str, '%Y-%m-%d').date()

            if start_date <= file_date <= end_date:
                file_path = os.path.join(self.directory, filename)
                # Use NumPy to read the CSV file
                data = np.loadtxt(file_path, delimiter=',', usecols=self.columns, skiprows=1)
                # print(f"Reading file: {filename}, Data shape: {data.shape}")  # Debugging print
                all_data.append(data)

        # Concatenate all data arrays
        combined_data = np.vstack(all_data)
        # print(f"Shape of the combined data: {combined_data.shape}")
        return combined_data

    def __len__(self):
        return len(self.data) - self.input_window_size - self.target_window_size + 1

    def __getitem__(self, idx):
        if idx + self.input_window_size + self.target_window_size > len(self.data):
            raise IndexError("Index out of bounds")

        window_data = self.data[idx:idx + self.input_window_size + self.target_window_size]
        open_prices = window_data[:, 0]  # Open prices column
        close_prices = window_data[:, 1]  # Close prices column
        percentage_changes = ((close_prices - open_prices) * 100 / open_prices)
        input_data = torch.tensor(percentage_changes[:self.input_window_size], dtype=torch.float32)
        target_data = torch.tensor(percentage_changes[self.input_window_size:], dtype=torch.float32)
        return idx, (input_data, target_data)
    

def change_to_binary(tensor):
    """
    Convert a tensor's positive values to 1 and negative values to 0 for binary classification.

    Args:
    tensor (torch.Tensor): A tensor containing numeric values.

    Returns:
    torch.Tensor: A tensor where all positive values are replaced with 1 and negative values with 0.
    """
    binary_tensor = torch.where(tensor > 0, 1., 0.)
    return binary_tensor

#----------------------------------------------------------------------
class DynamicMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=16, hidden_nodes=None, dropout_rate=dropout_rate):
        super(DynamicMLP, self).__init__()

        if hidden_nodes is None:
            hidden_nodes = input_size

        layers = []

        # Input layer with Batch Normalization
        layers.append(nn.Linear(input_size, hidden_nodes))
        layers.append(nn.BatchNorm1d(hidden_nodes))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            layers.append(nn.BatchNorm1d(hidden_nodes))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_nodes, output_size))

        # Combine all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

#----------------------------------------------------------------------
# Example usage
train_dataset = PriceDataset('BTCUSDT', '1m', '2021-03-01', '2021-05-31', input_window_size=input_size, target_window_size=target_size)
test_dataset = PriceDataset('BTCUSDT', '1m', '2023-01-01', '2023-03-01', input_window_size=input_size, target_window_size=target_size)
print(train_dataset.__len__())
print(test_dataset.__len__())

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 학습 데이터 로더
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   # 테스트 데이터 로더


# 모델 인스턴스 생성
model = DynamicMLP(input_size=input_size, output_size=target_size)
model = model.to(device)


# 손실 함수 및 옵티마이저
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



#----------------------------------------------------------------------
# Print dataset sizes and DataLoader tensor sizes, lengths, shapes, and first 5 examples
print("Train Dataset Length:", len(train_dataset))
print("Test Dataset Length:", len(test_dataset))

for i, (indices, (input_data, target_data)) in enumerate(train_loader):
    if i >= 5:
        break
    print(f"Batch {i+1}:")
    print("Indices:", indices)
    print("Input Data Shape:", input_data.shape)
    print("Target Data Shape:", target_data.shape)
    print("Input Data:", input_data)
    print("Target Data:", change_to_binary(target_data))

#----------------------------------------------------------------------
def test_first_batches(model, data_loader, criterion, device, num_batches=1):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i, (_, (input_data, target_data)) in enumerate(data_loader):
            if i >= num_batches:
                break  # Only test the first num_batches

            # Move data to the device directly
            input_data = input_data.to(device)
            target_data = target_data.to(device)

            # Model prediction
            output = model(input_data)

            # Calculate loss
            loss = criterion(output, change_to_binary(target_data))

            # Print batch information
            print(f"Batch {i + 1}/{num_batches}")
            print(f"Input Data: {input_data}")
            print(f"Target Data: {change_to_binary(target_data)}")
            print(f"Output Data: {output}")
            print(f"Loss: {loss.item()}\n")

# Test the function with your model, data loaders, criterion, and device
print("Testing on Training Data:")
test_first_batches(model, train_loader, criterion, device, num_batches=20)

print("Testing on Test Data:")
test_first_batches(model, test_loader, criterion, device, num_batches=20)


#----------------------------------------------------------------------
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for i, (batch_indices, (input_data, target_data)) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Data dimension change and move to device
        input_data = input_data.to(device)  # Adjust dimensions as needed
        target_data = target_data.to(device)  # Ensure this matches your model's output

        # Model prediction
        output = model(input_data)
        target_data = change_to_binary(target_data)

        # Loss calculation and backpropagation
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Average loss per batch
    avg_loss = total_loss / len(train_loader)
    print(f"Training: Average Loss: {avg_loss}")
    return avg_loss

def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0

    for i, (batch_indices, (input_data, target_data)) in enumerate(test_loader):
        # Data dimension change and move to device
        input_data = input_data.to(device)  # Adjust dimensions as needed
        target_data = target_data.to(device)  # Ensure this matches your model's output

        # Model prediction
        output = model(input_data)
        target_data = change_to_binary(target_data)

        # Loss calculation
        loss = criterion(output, target_data)
        total_loss += loss.item()

    # Average loss per batch
    avg_loss = total_loss / len(test_loader)
    print(f"Testing: Average Loss: {avg_loss}")
    return avg_loss


#----------------------------------------------------------------------
def create_model_filename(input_size, target_size, dropout_rate):
    return f"model_input{input_size}_target{target_size}_dropout{dropout_rate}.pth"

# 모델 파일 경로
model_file_path = f"model/final v3/{create_model_filename(input_size, target_size, dropout_rate)}"

# 모델 파일이 존재하는지 확인하고, 존재할 경우 모델 로드
if os.path.isfile(model_file_path):
    model.load_state_dict(torch.load(model_file_path))
    print("Pre-trained model loaded.")
else:
    print("No pre-trained model found. Initializing a new model.")

# 학습 과정
train_losses = []
test_losses = []
best_test_loss = float('inf')

# Define the directory for saving figures
figures_directory = "figures/final v3"
model_name = create_model_filename(input_size, target_size, dropout_rate=dropout_rate).replace('.pth', '')  # Use the generated model name without the file extension
model_figures_directory = os.path.join(figures_directory, model_name)

# Create the directories if they do not exist
os.makedirs(model_figures_directory, exist_ok=True)

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = test(model, test_loader, criterion)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    # Update best test loss and save model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), model_file_path)  # Corrected this line
        print(f"Model saved as '{model_file_path}' at Epoch {epoch + 1}")

    # Print epoch results
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}")

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
    plt.plot(range(1, epoch + 2), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()

    # Save and show the figure
    figure_path = os.path.join(model_figures_directory, f"Epoch_{epoch+1}.png")
    plt.savefig(figure_path)
    plt.show()
    plt.clf()


