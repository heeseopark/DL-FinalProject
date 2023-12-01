#----------------------------------------------------------------------
#----------------------------------------------------------------------
import torch
import pandas as pd
from datetime import datetime as dt
from torch.utils.data import Dataset, DataLoader
import os

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
        return torch.tensor(percentage_changes.values, dtype=torch.float64)



#----------------------------------------------------------------------
import torch
import torch.nn as nn
import math

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
        
        # Using the last time step's output
        output = torch.sigmoid(self.out(output[-1]))
        
        return output


#----------------------------------------------------------------------
import torch.optim as optim

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
num_epochs = 10

#----------------------------------------------------------------------
# Example usage
train_dataset = PriceDataset('BTCUSDT', '1m', '2021-03-01', '2021-12-01', input_window_size=input_size, target_window_size=target_size)
test_dataset = PriceDataset('BTCUSDT', '1m', '2022-01-01', '2022-01-30', input_window_size=input_size, target_window_size=target_size)


# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 학습 데이터 로더
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   # 테스트 데이터 로더


# 모델 인스턴스 생성
model = TimeSeriesTransformerModel(num_features, d_model, n_heads, num_encoder_layers, d_ff, dropout_rate, lstm_hidden_size, num_lstm_layers)

# 손실 함수 및 옵티마이저
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def change_percentage_tensor_to_binary(tensor):
    # 양수인 요소를 True로, 그렇지 않은 요소를 False로 변환
    binary_tensor = tensor > 0
    # Boolean 텐서를 정수형(0과 1)으로 변환
    return binary_tensor.float()

#----------------------------------------------------------------------
# 데이터셋 길이 확인
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# 첫 번째 데이터셋 샘플 확인
print("First sample in train dataset:", train_dataset[0])
print("First sample in test dataset:", test_dataset[0])

print("Train dataset shape:", train_dataset[0].shape)
print("Test dataset shape:", test_dataset[0].shape)

#----------------------------------------------------------------------
def print_first_batch_info(loader, name):
    for batch in loader:
        print(f"First batch in {name} DataLoader - Data shape: {batch.shape}")
        break

print_first_batch_info(train_loader, "train")
print_first_batch_info(test_loader, "test")

def print_first_batch_data(loader, name):
    for batch in loader:
        # 배치 데이터를 numpy 배열로 변환하여 출력
        print(f"First batch data in {name} DataLoader:")
        print(batch.numpy())  # 텐서를 numpy 배열로 변환
        break

# 첫 번째 배치의 데이터 값 확인
print_first_batch_data(train_loader, "train")
print_first_batch_data(test_loader, "test")


#----------------------------------------------------------------------
# train_dataset과 test_dataset의 크기 출력
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# train_dataset에서 첫 번째 예시 출력
print("First example in train dataset:")
print(f"Data: {next(iter(train_dataset))}")
print(f"Shape: {next(iter(train_dataset)).shape}")

# test_dataset에서 첫 번째 예시 출력
print("First example in train dataset:")
print(f"Data: {next(iter(test_dataset))}")
print(f"Shape: {next(iter(test_dataset)).shape}")

#----------------------------------------------------------------------
# train_loader와 test_loader의 window 개수 계산
train_windows = len(train_loader)
test_windows = len(test_loader)
print(f"Train windows: {train_windows}")
print(f"Test windows: {test_windows}")

# 예시 데이터와 해당 데이터의 shape 출력
for i, data in enumerate(train_loader):
    print(f"Train batch {i} data shape: {data.shape}")
    if i == 0:  # 첫 번째 배치만 출력하고 루프 종료
        break

for i, data in enumerate(test_loader):
    print(f"Test batch {i} data shape: {data.shape}")
    if i == 0:  # 첫 번째 배치만 출력하고 루프 종료
        break

    # 예시 데이터와 해당 데이터의 shape 출력
for i, data in enumerate(train_loader):
    print(f"Train batch {i} data: {data}")
    if i == 0:  # 첫 번째 배치만 출력하고 루프 종료
        break

for i, data in enumerate(test_loader):
    print(f"Train batch {i} data: {data}")
    if i == 0:  # 첫 번째 배치만 출력하고 루프 종료
        break

#----------------------------------------------------------------------
change_percentage_tensor_to_binary(train_dataset[15])

#----------------------------------------------------------------------
# 학습 루프
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for data in train_loader:
        # 입력과 타겟을 분리
        inputs = data[:, :input_size]  # 처음 10개 값
        targets = data[:, input_size:]  # 다음 5개 값

        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(-1))  # 모델에 맞게 차원 조정 필요
        loss = criterion(outputs, targets.unsqueeze(-1))  # 차원 조정 필요
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {average_train_loss:.4f}")

# 테스트 루프
model.eval()
total_test_loss = 0
with torch.no_grad():
    for data in test_loader:
        inputs = data[:, :input_size]
        targets = data[:, input_size:]

        outputs = model(inputs.unsqueeze(-1))
        loss = criterion(outputs, targets.unsqueeze(-1))
        total_test_loss += loss.item()

    average_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {average_test_loss:.4f}")


