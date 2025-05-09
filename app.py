
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

        labels = [f"{model.model.names[int(d.cls)]} ({d.conf:.2f})" for d in result.boxes.data]

        return render_template('result.html', image=filename, labels=labels)

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render подаёт переменную PORT
    app.run(host='0.0.0.0', port=port)
