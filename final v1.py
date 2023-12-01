#----------------------------------------------------------------------

import torch
import torch.nn as nn
import math
import pandas as pd
from datetime import datetime as dt
from torch.utils.data import Dataset, DataLoader
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
seed = 42  # choose any seed you prefer
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#----------------------------------------------------------------------
class PriceDataset(Dataset):
    def __init__(self, item, timespan, start_date_str, end_date_str, input_window_size, target_window_size):
        self.directory = f'C:/Github/DL-FinalProject/csvfiles/{item}/'
        self.input_window_size = input_window_size
        self.target_window_size = target_window_size
        self.columns = [1, 4]
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
                df = pd.read_csv(file_path, usecols=self.columns)
                all_data.append(df)

        return pd.concat(all_data, ignore_index=True)

    def __len__(self):
        return len(self.data) - self.input_window_size - self.target_window_size + 1

    def __getitem__(self, idx):
        if idx + self.input_window_size + self.target_window_size > len(self.data):
            raise IndexError("Index out of bounds")

        window_data = self.data.iloc[idx:idx + self.input_window_size + self.target_window_size]
        open_prices = window_data.iloc[:, 0]  # Assuming 1st column is 'open' prices
        close_prices = window_data.iloc[:, 1]  # Assuming 4th column is 'close' prices
        percentage_changes = ((close_prices - open_prices) * 100 / open_prices)
        return torch.tensor(percentage_changes.values, dtype=torch.float32)

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
    def __init__(self, num_features, d_model, n_heads, num_encoder_layers, d_ff, dropout_rate, lstm_hidden_size, num_lstm_layers):
        super(TimeSeriesTransformerModel, self).__init__()
        self.d_model = d_model

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True)
        
        # Linear layer to transform LSTM output to match Transformer d_model size
        self.linear1 = nn.Linear(lstm_hidden_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Output layer
        self.out = nn.Linear(d_model, 1)

    def forward(self, src):
        # LSTM layer
        lstm_out, _ = self.lstm(src)

        # Transform LSTM output to match Transformer d_model size
        src = self.linear1(lstm_out) * math.sqrt(self.d_model)

        # Positional Encoding and Transformer
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        # Adjusting the output layer to return target_size values
        output = torch.sigmoid(self.out(output)).squeeze()

        return output

#----------------------------------------------------------------------
def change_percentage_tensor_to_binary(tensor):
    # 양수인 요소를 True로, 그렇지 않은 요소를 False로 변환
    binary_tensor = tensor > 0
    # Boolean 텐서를 정수형(0과 1)으로 변환
    return binary_tensor.float()

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        
        # 데이터에 차원 추가
        data = batch.unsqueeze(-1)
        data = data.to(device)
        
        # 타겟 생성 (예를 들어, 마지막 target_size 값)
        target = change_percentage_tensor_to_binary(data[:, -target_size:])
        target = target.to(device)
        
        # 모델 예측
        output = model(data)
        
        # 손실 계산 및 역전파
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            # 데이터에 차원 추가
            data = batch.unsqueeze(-1)
            data = data.to(device)

            # 타겟 생성
            target = change_percentage_tensor_to_binary(data[:, -target_size:])
            target = target.to(device)
            # 모델 예측
            output = model(data)
            
            # 손실 계산
            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(test_loader)


#----------------------------------------------------------------------
# hyperparameters for dataset and dataloader
input_size = 10
target_size = 5
batch_size = 32

# hyperparameters for model
num_features = 1  # 입력 시퀀스의 특징 수 (예: open, close 가격)
d_model = 64
n_heads = 2
num_encoder_layers = 2
d_ff = 256
dropout_rate = 0.1
lstm_hidden_size = 128
num_lstm_layers = 2

# hyperparameters for training, testing
learning_rate = 0.001
num_epochs = 2

#----------------------------------------------------------------------
# Example usage
train_dataset = PriceDataset('BTCUSDT', '1m', '2021-03-01', '2021-12-01', input_window_size=input_size, target_window_size=target_size)
test_dataset = PriceDataset('BTCUSDT', '1m', '2022-01-01', '2022-01-30', input_window_size=input_size, target_window_size=target_size)


# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 학습 데이터 로더
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   # 테스트 데이터 로더


# 모델 인스턴스 생성
model = TimeSeriesTransformerModel(num_features, d_model, n_heads, num_encoder_layers, d_ff, dropout_rate, lstm_hidden_size, num_lstm_layers)
model = model.to(device)


# 손실 함수 및 옵티마이저
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#----------------------------------------------------------------------
# 모델 파일 경로
model_file_path = "best_model.pth"

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

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = test(model, test_loader, criterion)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}")

    # 테스트 손실이 이전 최소값보다 낮을 때만 모델 저장
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), model_file_path)
        print(f"Model saved at Epoch {epoch + 1}")

    # 학습 과정 시각화
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.show()


