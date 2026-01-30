from flask import Flask, render_template, request, send_from_directory, send_file
from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime
from fpdf import FPDF
import pandas as pd

app = Flask(__name__)

# ------------------- Директории -------------------
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
HISTORY_FILE = "history/history.json"

# Создаём директории, если их нет
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

# ------------------- Модель -------------------
model = YOLO("yolov8n.pt")  # Предобученная модель YOLOv8

# ------------------- Функция детекции -------------------
def detect_ball(image_path):
    image = cv2.imread(image_path)
    results = model(image)

    ball_detected = False
    confidence = 0
    bbox = None

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 32:  # sports ball
                ball_detected = True
                confidence = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = [x1, y1, x2, y2]
                # Нарисовать bbox на изображении
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Ball {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break

    filename = os.path.basename(image_path)
    result_path = os.path.join(RESULT_DIR, filename)
    cv2.imwrite(result_path, image)

    # Запись в историю
    record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": filename,
        "ball_detected": ball_detected,
        "confidence": round(confidence, 2),
        "bbox": bbox
    }

    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return filename, record

# ------------------- Статистика -------------------
def get_statistics():
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    total = len(history)
    found = sum(1 for r in history if r["ball_detected"])
    not_found = total - found
    avg_confidence = round(sum(r["confidence"] for r in history) / total, 2) if total > 0 else 0

    return {
        "total_images": total,
        "found": found,
        "not_found": not_found,
        "avg_confidence": avg_confidence
    }

# ------------------- PDF отчёт -------------------
@app.route("/download_pdf")
def download_pdf():
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    pdf = FPDF()
    pdf.add_page()

    # Register fonts
    pdf.add_font("DejaVu", "", "dejavu-sans/DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "dejavu-sans/DejaVuSans-Bold.ttf", uni=True)

    # Header
    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 10, "Отчёт по анализу положения футбольного мяча", ln=True, align="C")
    pdf.ln(10)

    # History
    pdf.set_font("DejaVu", "", 12)
    for record in history:
        status = "Мяч найден" if record["ball_detected"] else "Мяч не найден"
        explanation = (
            "Модель определила местоположение мяча с высокой точностью."
            if record["ball_detected"] else "Мяч не найден."
        )
        pdf.cell(0, 6, f"{record['date']} | {record['image']} | {status} | {record['confidence']*100:.1f}%", ln=True)
        pdf.multi_cell(0, 6, f"Пояснение: {explanation}")
        pdf.ln(2)

    # Summary
    total_images = len(history)
    found = sum(1 for r in history if r["ball_detected"])
    not_found = total_images - found
    avg_confidence = sum(r["confidence"] for r in history)/total_images if total_images else 0

    pdf.ln(5)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 6, f"Всего изображений: {total_images}", ln=True)
    pdf.cell(0, 6, f"Мяч найден: {found}", ln=True)
    pdf.cell(0, 6, f"Мяч не найден: {not_found}", ln=True)
    pdf.cell(0, 6, f"Средняя уверенность модели: {avg_confidence*100:.1f}%", ln=True)

    pdf_file = "report.pdf"
    pdf.output(pdf_file)
    return send_file(pdf_file, as_attachment=True)

# ------------------- Excel отчёт -------------------
@app.route("/download_excel")
def download_excel():
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    df = pd.DataFrame(history)
    total_images = len(history)
    found = sum(1 for r in history if r["ball_detected"])
    not_found = total_images - found
    avg_confidence = sum(r["confidence"] for r in history)/total_images if total_images else 0

    summary = pd.DataFrame([{
        "date": "Итог",
        "image": "",
        "ball_detected": found,
        "confidence": avg_confidence,
        "explanation": f"Всего изображений: {total_images}, Мяч найден: {found}, Мяч не найден: {not_found}, Средняя уверенность модели: {avg_confidence*100:.1f}%"
    }])

    df = pd.concat([df, summary], ignore_index=True)
    excel_file = "report.xlsx"
    df.to_excel(excel_file, index=False)
    return send_file(excel_file, as_attachment=True)

# ------------------- Показ изображений -------------------
@app.route("/results/<filename>")
def show_result_image(filename):
    return send_from_directory(RESULT_DIR, filename)

# ------------------- Главная страница -------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = os.path.join(UPLOAD_DIR, file.filename)
            file.save(path)
            result = detect_ball(path)

    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    stats = get_statistics()
    return render_template("index.html", result=result, history=history, stats=stats)

# ------------------- Запуск -------------------
if __name__ == "__main__":
    app.run(debug=True)
