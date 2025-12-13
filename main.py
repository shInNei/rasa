import pandas as pd
import json
import os
import torch
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# --- CẤU HÌNH ---
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv" 
MODEL_PATH = "my-smart-home-model"

def load_and_prepare_data(train_path, test_path):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Không tìm thấy file dataset.")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # 1. Lọc dữ liệu rác
    df_train = df_train[df_train['label_str'] != 'label_str']
    df_test = df_test[df_test['label_str'] != 'label_str']
    df_train = df_train.dropna(subset=['label_str', 'text'])
    df_test = df_test.dropna(subset=['label_str', 'text'])

    # 2. Tạo Map từ tập TRAIN (Đây là bộ chuẩn)
    unique_labels = sorted(df_train['label_str'].unique().tolist())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    
    print(f"-> Đã load {len(df_train)} dòng train, {len(df_test)} dòng test.")
    print("-> Danh sách nhãn:", label2id)

    # 3. Map dữ liệu
    df_train['label'] = df_train['label_str'].map(label2id)
    
    # Lưu ý: Nếu test có nhãn lạ chưa từng train, ta sẽ bỏ qua để tránh lỗi
    df_test['label'] = df_test['label_str'].map(label2id)
    df_test = df_test.dropna(subset=['label']) # Bỏ dòng nhãn lạ
    
    return df_train, df_test, label2id, id2label

# --- MAIN ---
print("=== 1. ĐANG TẢI DỮ LIỆU ===")
df_train, df_test, label2id, id2label = load_and_prepare_data(TRAIN_FILE, TEST_FILE)

train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

print("\n=== 2. ĐANG TẢI MODEL ===")
# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 DEVICE: {device.upper()}")

model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device=device
)

# Tăng epochs lên 3 để model học kỹ hơn các nhãn con
args = TrainingArguments(
    batch_size=16,
    num_epochs=2, # <--- TĂNG LÊN 3
    loss=CosineSimilarityLoss,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    column_mapping={"text": "text", "label": "label"}
)

print("\n=== 3. BẮT ĐẦU TRAIN ===")
trainer.train()

print("\n=== 4. ĐÁNH GIÁ ===")
preds = model(df_test['text'].tolist())
y_pred = preds.tolist()
y_true = df_test['label'].tolist()

acc = accuracy_score(y_true, y_pred)
print(f"ĐỘ CHÍNH XÁC: {acc * 100:.2f}%")

print("\n--- Báo cáo chi tiết ---")
# --- FIX LỖI REPORT TRIỆT ĐỂ ---
# Lấy danh sách ID của TOÀN BỘ nhãn (từ tập train)
all_labels_ids = list(id2label.keys())
all_target_names = list(id2label.values())

# Ép hàm report in ra đủ danh sách này, nếu tập test thiếu thì nó ghi 0
print(classification_report(
    y_true, 
    y_pred, 
    labels=all_labels_ids, 
    target_names=all_target_names, 
    zero_division=0
))

print("\n--- Các câu sai ---")
for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
        print(f"Câu: '{df_test.iloc[i]['text']}'")
        print(f"   Thực tế: {id2label[y_true[i]]} | Máy đoán: {id2label[y_pred[i]]}")

# Lưu model
model.save_pretrained(MODEL_PATH)
with open(f"{MODEL_PATH}/label_map.json", "w", encoding="utf-8") as f:
    json.dump(id2label, f, ensure_ascii=False, indent=4)
print("\nĐã lưu model xong.")