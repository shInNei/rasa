import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from setfit import SetFitModel

# --- CẤU HÌNH ---
MODEL_PATH = "my-smart-home-model"

# Khởi tạo App
app = FastAPI(title="Smart Home AI Service")

# Biến toàn cục để lưu model
ai_model = None
id2label = {}

# Class quy định dữ liệu đầu vào (Input Schema)
class CommandRequest(BaseModel):
    text: str

# --- HÀM HỖ TRỢ ---
def extract_number(text):
    """Tìm số trong câu (dùng cho chỉnh level/nhiệt độ)"""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None

# --- SỰ KIỆN KHỞI ĐỘNG (Load Model 1 lần duy nhất) ---
@app.on_event("startup")
def load_resources():
    global ai_model, id2label
    print("⏳ Đang tải model AI lên RAM...")
    
    try:
        # 1. Load Model SetFit
        ai_model = SetFitModel.from_pretrained(MODEL_PATH)
        
        # 2. Load Label Map
        with open(f"{MODEL_PATH}/label_map.json", "r", encoding="utf-8") as f:
            loaded_map = json.load(f)
            # JSON lưu key là string "0", cần chuyển về int 0
            id2label = {int(k): v for k, v in loaded_map.items()}
            
        print("✅ AI Service đã sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi tải model: {e}")

# --- API ENDPOINT (Cổng giao tiếp) ---
@app.post("/predict")
async def predict_intent(req: CommandRequest):
    if not ai_model:
        raise HTTPException(status_code=500, detail="Model chưa được load")

    # 1. Dự đoán Intent (Ý định)
    # Model trả về Tensor, cần lấy item() để ra số int
    pred_id = ai_model([req.text])[0].item()
    intent = id2label.get(pred_id, "UNKNOWN")

    # 2. Trích xuất Entity (Số) nếu có
    # Chỉ tìm số nếu intent liên quan đến việc chỉnh mức độ
    entity_value = None
    if "SET_LEVEL" in intent or "THRESHOLD" in intent or "TEMP" in intent:
        entity_value = extract_number(req.text)

    # 3. Trả về kết quả JSON
    return {
        "text": req.text,
        "intent": intent,
        "entity_value": entity_value,
        "confidence": "High" # SetFit bản hiện tại chưa hỗ trợ trả confidence score trực tiếp dễ dàng, tạm để High
    }

# Để chạy: uvicorn ai_service:app --reload --port 8000