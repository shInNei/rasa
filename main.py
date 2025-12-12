import pandas as pd
import json
import os
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.metrics import accuracy_score, classification_report

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv" 
MODEL_PATH = "my-smart-home-model"

def load_and_prepare_data(train_path, test_path):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Không tìm thấy file dataset.csv hoặc test_dataset.csv")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"-> Đã load {len(df_train)} dòng train và {len(df_test)} dòng test.")

    unique_labels = df_train['label_str'].unique().tolist()
    unique_labels.sort()
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    
    print("-> Danh sách nhãn:", label2id)

    df_train['label'] = df_train['label_str'].map(label2id)
    
    df_test['label'] = df_test['label_str'].map(label2id)
    df_test = df_test.dropna(subset=['label'])
    df_test['label'] = df_test['label'].astype(int)

    return df_train, df_test, label2id, id2label


print("=== 1. ĐANG TẢI DỮ LIỆU ===")
df_train, df_test, label2id, id2label = load_and_prepare_data(TRAIN_FILE, TEST_FILE)

train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

print("\n=== 2. ĐANG TẢI MODEL NỀN ===")
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

args = TrainingArguments(
    batch_size=16,
    num_epochs=1, 
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

print("\n=== 4. ĐÁNH GIÁ TRÊN TẬP TEST ===")
preds = model(df_test['text'].tolist())

y_pred = preds.tolist()
y_true = df_test['label'].tolist()
acc = accuracy_score(y_true, y_pred)
print(f"ĐỘ CHÍNH XÁC (Accuracy): {acc * 100:.2f}%")

print("\n--- Báo cáo chi tiết ---")
target_names = [id2label[i] for i in range(len(id2label))]
print(classification_report(y_true, y_pred, target_names=target_names))

print("\n--- Các câu máy đoán sai ---")
for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
        print(f"Câu: '{df_test.iloc[i]['text']}'")
        print(f"   Thực tế: {id2label[y_true[i]]} | Máy đoán: {id2label[y_pred[i]]}")
        print("-" * 20)

print("\n=== 5. LƯU MODEL ===")
model.save_pretrained(MODEL_PATH)

with open(f"{MODEL_PATH}/label_map.json", "w", encoding="utf-8") as f:
    json.dump(id2label, f, ensure_ascii=False, indent=4)

print(f"Đã lưu model và label_map vào thư mục '{MODEL_PATH}'")