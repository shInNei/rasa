# Prerequisite:
- Python version 3.12
(Các version cũ hơn hoặc mới hơn sẽ gặp bug)
python -m venv venv
# Lưu ý nếu không có python version 3.12
1. Tải về python version 3.12 tại đây: https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe
2. Tạo env bằng lệnh sau: py -3.12 -m venv venv   

3. .\venv\Scripts\activate  
pip install -r requirements.txt      
(lệnh trên lệnh cài đặt)
4. uvicorn demo:app --reload --port 8000 (lệnh chạy) 
(trước khi chạy lệnh cuối nhớ tải này về rồi giải nén folder này vào cùng folder chứa code, ko có để folder lồng folder lúc giải nén
https://drive.google.com/file/d/1klYfvn5pJuDoKVBW5VWPyuy2IwTSDiQj/view?usp=sharing)