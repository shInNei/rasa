import json
import os
from setfit import SetFitModel
from sklearn.metrics import accuracy_score

MODEL_PATH = "my-smart-home-model"

challenge_data = [
    {"text": "anh muốn bật đèn lên ngay lập tức", "label": "LIGHT_ON"},
    {"text": "bot ơi làm ơn mở cái đèn giùm tao", "label": "LIGHT_ON"},
    {"text": "chị không thấy đường em bật điện lên đi", "label": "LIGHT_ON"},
    {"text": "này trợ lý ảo, kích hoạt hệ thống chiếu sáng", "label": "LIGHT_ON"},
    {"text": "tối om rồi bật bóng tuýp lên coi", "label": "LIGHT_ON"},

    {"text": "tao đi ngủ đây tắt đèn nha", "label": "LIGHT_OFF"},
    {"text": "chói mắt quá tắt bớt điện đi em", "label": "LIGHT_OFF"},
    {"text": "mày tắt cái bóng đèn đó ngay cho tao", "label": "LIGHT_OFF"},
    {"text": "ra khỏi phòng rồi ngắt điện giùm", "label": "LIGHT_OFF"},
    {"text": "bot tắt đèn", "label": "LIGHT_OFF"},

    {"text": "nóng chảy mỡ rồi bật quạt lên đi trời", "label": "FAN_ON"},
    {"text": "anh hai muốn khởi động quạt số 3", "label": "FAN_ON"},
    {"text": "làm ơn cho chút gió mát đi bạn ơi", "label": "FAN_ON"},
    {"text": "ê cu bật cái máy quạt coi", "label": "FAN_ON"},
    {"text": "phòng bí quá mở quạt trần lên", "label": "FAN_ON"},

    {"text": "lạnh teo bugi rồi tắt quạt đi", "label": "FAN_OFF"},
    {"text": "ồn ào quá dừng quạt lại ngay", "label": "FAN_OFF"},
    {"text": "em tắt cái máy gió giùm anh nha", "label": "FAN_OFF"},
    {"text": "không cần quạt nữa đâu tắt đi", "label": "FAN_OFF"},
    {"text": "stop cái quạt trần lại hộ cái", "label": "FAN_OFF"},

    {"text": "chào em yêu", "label": "NONE"},
    {"text": "mở cửa ra cho thoáng", "label": "NONE"}, 
    {"text": "bật tivi lên xem đá banh", "label": "NONE"}, 
    {"text": "anh đói bụng quá", "label": "NONE"},
    {"text": "gọi điện cho mẹ anh đi", "label": "NONE"}
]

def run_challenge():
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy folder '{MODEL_PATH}'. Bạn đã train chưa?")
        return

    print(f"Đang tải model từ '{MODEL_PATH}'...")
    model = SetFitModel.from_pretrained(MODEL_PATH)
    
    try:
        with open(f"{MODEL_PATH}/label_map.json", "r", encoding="utf-8") as f:
            id2label = json.load(f)
            id2label = {int(k): v for k, v in id2label.items()}
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file label_map.json")
        return

    inputs = [item["text"] for item in challenge_data]
    expected_labels = [item["label"] for item in challenge_data]

    print(f"Đang dự đoán {len(inputs)} câu khó...")
    preds = model(inputs)

    correct_count = 0
    print("\n" + "="*85)
    print(f"{'CÂU NÓI (INPUT)':<45} | {'THỰC TẾ':<10} | {'DỰ ĐOÁN':<10} | {'KẾT QUẢ':<5}")
    print("="*85)

    for i, text in enumerate(inputs):
        pred_id = preds[i].item() 
        pred_label_str = id2label.get(pred_id, "UNKNOWN")
        true_label_str = expected_labels[i]

        is_correct = pred_label_str == true_label_str
        if is_correct:
            correct_count += 1
            status = "Correct"
        else:
            status = "InCorrect"

        print(f"{text:<45} | {true_label_str:<10} | {pred_label_str:<10} | {status}")

    # 6. Tổng kết
    accuracy = (correct_count / len(inputs)) * 100
    print("="*85)
    print(f"KẾT QUẢ: Đúng {correct_count}/{len(inputs)} câu")
    print(f"ĐỘ CHÍNH XÁC: {accuracy:.2f}%")
    
    if accuracy >= 80:
        print("=> ĐÁNH GIÁ: Model hoạt động TỐT với các câu có chủ ngữ/vị ngữ phức tạp! 🚀")
    else:
        print("=> ĐÁNH GIÁ: Cần train thêm các mẫu câu dài để cải thiện.")

if __name__ == "__main__":
    run_challenge()