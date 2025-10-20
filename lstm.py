import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from google.colab import drive

drive.mount('/content/drive')

# 설정 변수
TRAIN_BASE_PATH = '/content/drive/MyDrive/colab_data'
VALIDATION_BASE_PATH = '/content/drive/MyDrive/colab_data/Validation'
ONNX_SAVE_PATH = '/content/drive/MyDrive/colab_data/lstm_biometric_model_6steps_final_v3.onnx'

N_STEPS = 6
FEATURE_COLUMNS = ['Heartrate', 'SPO2', 'Walking_steps', 'Caloricexpenditure']
TARGET_COLUMN = 'label'
# 하이퍼파라미터
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5
BATCH_SIZE = 128

N_FEATURES = len(FEATURE_COLUMNS)
HIDDEN_SIZE = 64
NUM_LAYERS = 1
N_HEADS = 4

# 데이터 로드 함수
def load_and_merge_data(base_path):
    all_files = [f for f in os.listdir(base_path) if f.startswith('L_A00') and f.endswith('.csv')]
    list_df = [pd.read_csv(os.path.join(base_path, f)) for f in all_files]
    if not list_df:
        print(f"경로에 파일이 없습니다: {base_path}")
        return pd.DataFrame()
    df_merged = pd.concat(list_df, ignore_index=True)
    return df_merged

# PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y, n_steps):
        X_seq, Y_seq = [], []
        for i in range(len(X) - n_steps + 1):
            X_seq.append(X[i:i + n_steps])
            Y_seq.append(Y[i + n_steps - 1])
        self.X = torch.tensor(np.array(X_seq), dtype=torch.float32)
        self.Y = torch.tensor(np.array(Y_seq), dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 수정된 모델
class BiLSTM_Attention_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_heads, n_classes, dropout_rate):
        super(BiLSTM_Attention_Model, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=n_heads,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout_rate)

        self.final_lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size // 2, 64)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(64, n_classes)

    def forward(self, x, inference=False):
        lstm_out, _ = self.bilstm(x)
        attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        x_attn = self.attn_norm(lstm_out + attn_output)
        x_attn = self.dropout(x_attn)

        lstm_final_out, (hn, cn) = self.final_lstm(x_attn)
        hn = hn.squeeze(0)

        x = self.fc1(hn)
        x = self.relu(x)
        x = self.dropout(x)

        logits = self.fc_out(x)
        if inference:
            return torch.softmax(logits, dim=1)
        else:
            return logits

# 데이터 로드
df_train = load_and_merge_data(TRAIN_BASE_PATH)
df_val = load_and_merge_data(VALIDATION_BASE_PATH)

le = LabelEncoder()
scaler = MinMaxScaler()
Y_train_encoded = le.fit_transform(df_train[TARGET_COLUMN])
Y_val_encoded = le.transform(df_val[TARGET_COLUMN])

class_weights_arr = compute_class_weight('balanced', classes=np.unique(Y_train_encoded), y=Y_train_encoded)
class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32)

X_train_scaled = scaler.fit_transform(df_train[FEATURE_COLUMNS])
X_val_scaled = scaler.transform(df_val[FEATURE_COLUMNS])

train_dataset = TimeSeriesDataset(X_train_scaled, Y_train_encoded, N_STEPS)
val_dataset = TimeSeriesDataset(X_val_scaled, Y_val_encoded, N_STEPS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

N_CLASSES = len(le.classes_)
print(f"클래스 개수: {N_CLASSES}")
print(f"라벨 매핑: {dict(zip(le.classes_, range(len(le.classes_))))}")
print(f"PyTorch Class Weights: {class_weights_tensor}")
print("-" * 30)

# 학습 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTM_Attention_Model(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, N_HEADS, N_CLASSES, DROPOUT_RATE).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

NUM_EPOCHS = 80
best_val_loss = float('inf')
patience = 10
patience_counter = 0

print("--- 모델 학습 시작 ---")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)  # inference=False (Softmax 없음)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    lr_scheduler.step(avg_val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], LR: {current_lr:.6f}, Loss: {avg_train_loss:.4f} - Acc: {train_accuracy:.2f}% - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_accuracy:.2f}%')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_pytorch_model_final_v3.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


#  ONNX 변환 
model.load_state_dict(torch.load('best_pytorch_model_final_v3.pth'))
model.to('cpu')
model.eval()

dummy_input = torch.randn(1, N_STEPS, N_FEATURES, dtype=torch.float32)
torch.onnx.export(
    model,
    (dummy_input, True),  # inference=True → Softmax 적용
    ONNX_SAVE_PATH,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"\n ONNX 모델 저장 완료: {ONNX_SAVE_PATH}")

scaler_min = scaler.min_
scaler_scale = scaler.scale_
scaler_max = scaler.data_max_

# 특징 이름
feature_names = FEATURE_COLUMNS
scaler_params = {}

for i, name in enumerate(feature_names):
    # Min-Max 스케일링 공식: scaled = (original - min) / (max - min)
    # scale_ = 1 / (max - min) 이므로, max = min + 1/scale_
    # 하지만, scaler.data_min_와 scaler.data_max_를 사용하는 것이 더 직관적
    
    # 데이터셋에 따라 min_과 data_min_이 다를 수 있지만,
    # fit_transform을 사용했으므로 data_min_ / data_max_를 사용하는 것이 가장 정확합니다.

    min_val = scaler.data_min_[i].item() # .item()으로 numpy float을 Python float으로 변환
    max_val = scaler.data_max_[i].item()
    
    scaler_params[name] = {
        "min": float(f'{min_val:.5f}'), # 소수점 5자리까지만 저장하여 정확도와 파일 크기 절충
        "max": float(f'{max_val:.5f}')
    }

# JavaScript 코드 형식으로 출력 (React에 복사/붙여넣기 위함)
print("\n" + "=" * 50)
print("React/JavaScript용 MinMaxScaler 파라미터")
print("=" * 50)
print("const scalerParams = {")
for name, params in scaler_params.items():
    print(f"    {name}: {{ min: {params['min']}, max: {params['max']} }},")
print("};")
print("=" * 50)
print("이 파라미터를 React 코드의 `scalerParams` 변수에 복사하여 사용하세요.")