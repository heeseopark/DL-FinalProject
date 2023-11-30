#----------------------------------------------------------------------
import torch
from torch.utils.data import DataLoader
from datetime import datetime as dt, timedelta
import pandas as pd
import os
import random
import numpy as np
import torch.nn as nn


# check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
seed = 42  # choose any seed you prefer
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#----------------------------------------------------------------------
# Dataset parameters and Lstm hyperparameters
window_size = 100 # lstm input size

input_window_size = 100

target_window_size = 10 # lstm output size

hidden_size = 1000

num_layers = 4

dropout = 0.1

#----------------------------------------------------------------------
class PriceDataset(torch.utils.data.Dataset):
    def __init__(self, item, timespan, start_date_str, end_date_str):
        self.directory = f'C:/Github/DL-FinalProject/csvfiles/{item}'
        self.item = item
        self.timespan = timespan
        start_date = dt.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = dt.strptime(end_date_str, '%Y-%m-%d').date()
        self.dates = [single_date.strftime("%Y-%m-%d") for single_date in self.daterange(start_date, end_date)]
        self.columns = [1, 4]  # Selecting open and close prices
        self.filenames = self.get_filenames()

    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + timedelta(n)

    def get_filenames(self):
        filenames = []
        for date in self.dates:
            filename = f"{self.directory}/{self.item}-{self.timespan}-{date}.csv"
            if os.path.exists(filename):
                filenames.append(filename)
        return filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        df = pd.read_csv(filename, usecols=self.columns, header=None)
        tensor = torch.tensor(df.values, dtype=torch.float)  # Return open and close prices
        return tensor


def sliding_window_percentage(batch):
    windows_percentage = []
    for tensor in batch:
        for i in range(tensor.shape[0] - input_window_size - target_window_size + 1):  # Create windows of size window_size
            window = tensor[i:i+input_window_size+target_window_size]
            pct_change = ((window[:, 1] - window[:, 0]) * 100 / window[:, 0])
            windows_percentage.append(pct_change)
    output_percentage = torch.stack(windows_percentage)

    return output_percentage

def sliding_window_binary(batch):
    windows_binary = []
    for tensor in batch:
        for i in range(tensor.shape[0] - input_window_size - target_window_size+ 1):  # Create windows of size window_size
            window = tensor[i:i+input_window_size+target_window_size]
            binary_change = (window[:, 1] > window[:, 0]).float()  # Calculate the binary change
            windows_binary.append(binary_change)
    output_binary = torch.stack(windows_binary)

    return output_binary

#----------------------------------------------------------------------
train_dataset = PriceDataset('BTCUSDT', '1m', '2021-03-01', '2023-04-30')
test_dataset = PriceDataset('ETHUSDT', '1m', '2021-03-01', '2023-04-30')


percentage_train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=sliding_window_percentage, shuffle=False, drop_last=True)
percentage_test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=sliding_window_percentage, shuffle=False, drop_last=True)

binary_train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=sliding_window_binary, shuffle=False, drop_last=True)
binary_test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=sliding_window_binary, shuffle=False, drop_last=True)

def count_total_windows(dataset, input_window_size, target_window_size):
    total_windows = 0
    for filename in dataset.filenames:
        df = pd.read_csv(filename, usecols=dataset.columns, header=None)
        # Calculate the number of windows in this file
        num_rows = len(df)
        if num_rows >= input_window_size + target_window_size:
            windows_in_file = num_rows - input_window_size - target_window_size + 1
            total_windows += windows_in_file
    return total_windows

# Example usage
total_train_windows = count_total_windows(train_dataset, input_window_size, target_window_size)
total_test_windows = count_total_windows(test_dataset, input_window_size, target_window_size)

print(f"Total windows in train dataset: {total_train_windows}")
print(f"Total windows in test dataset: {total_test_windows}")
