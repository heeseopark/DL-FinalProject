#----------------------------------------------------------------------
import torch
from torch.utils.data import DataLoader
from datetime import datetime as dt, timedelta
import pandas as pd
import os
import random
import numpy as np
import torch.nn as nn
import math
import matplotlib.pyplot as plt


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
input_window_size = 100
target_window_size = 10
batch_size = 32
learning_rate = 0.001
num_epochs = 10

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
        self.data = self.load_data()

    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + timedelta(n)

    def load_data(self):
        all_data = []
        for date in self.dates:
            filename = f"{self.directory}/{self.item}-{self.timespan}-{date}.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename, usecols=self.columns, header=None)
                all_data.append(df.values)
        return np.vstack(all_data)  # Concatenate all data into a single array

    def __len__(self):
        return len(self.data) - input_window_size

    def __getitem__(self, idx):
        window = self.data[idx:idx+input_window_size+1]  # +1 to include the next point for label
        features = torch.tensor(window[:-1], dtype=torch.float)  # All except last for input
        next_price = window[-1, 1]  # Close price of the next point
        last_price = window[-1, 0]  # Open price of the last point
        label = torch.tensor([next_price > last_price], dtype=torch.float)  # Binary label
        return features, label


#----------------------------------------------------------------------
train_dataset = PriceDataset('BTCUSDT', '1m', '2021-03-01', '2023-04-30')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

test_dataset = PriceDataset('ETHUSDT', '1m', '2021-03-01', '2023-04-30')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


#----------------------------------------------------------------------
def count_total_windows(dataset, input_window_size):
    num_rows = len(dataset.data)
    total_windows = num_rows - input_window_size if num_rows >= input_window_size + 1 else 0
    return total_windows

# Example usage
total_train_windows = count_total_windows(train_dataset, input_window_size)
total_test_windows = count_total_windows(test_dataset, input_window_size)

print(f"Total windows in train dataset: {total_train_windows}")
print(f"Total windows in test dataset: {total_test_windows}")


#----------------------------------------------------------------------
# 포지셔널 인코딩 클래스
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 시계열 트랜스포머 모델 클래스
class TimeSeriesTransformerModel(nn.Module):
    def __init__(self, num_features, d_model, n_heads, num_encoder_layers, d_ff, dropout_rate):
        super(TimeSeriesTransformerModel, self).__init__()
        self.d_model = d_model
        self.linear = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.out = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.linear(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = torch.sigmoid(self.out(output[-1]))  # 마지막 시점의 출력만 사용
        return output

#----------------------------------------------------------------------
# 모델 초기화
model = TimeSeriesTransformerModel(...)

# 손실 함수 및 최적화
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 에폭별 손실 추적을 위한 리스트
train_losses = []
test_losses = []

#----------------------------------------------------------------------
# 훈련 및 테스트 루프
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

    # 평균 손실 계산
    train_loss /= total_train_windows
    test_loss /= total_test_windows
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    # 그래프 업데이트
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Epoch vs Loss')
    plt.show()

