
from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
from PIL import Image
import os
import uuid

app = Flask(__name__)
model = YOLO('best.pt')  # Убедитесь, что best.pt загружен в корень проекта
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        results = model(filepath)
        result = results[0]
        result.save(filename=os.path.join(RESULT_FOLDER, filename))

        # Получаем списки классов и доверий из result.boxes
        cls_indices = result.boxes.cls.tolist()     # e.g. [0, 4, ...]
        confidences = result.boxes.conf.tolist()    # e.g. [0.85, 0.76, ...]
        
        # Формируем человеко-читаемые метки
        labels = [
            f"{model.model.names[int(cls_idx)]} ({conf:.2f})"
            for cls_idx, conf in zip(cls_indices, confidences)
        ]        

        return render_template('result.html', image=filename, labels=labels)

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render подаёт переменную PORT
    app.run(host='0.0.0.0', port=port)
