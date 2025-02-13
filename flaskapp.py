from flask import Flask, request, render_template
from PIL import Image
import io
import os
from pipeline.predict import predict_class, class_names

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html', result=None, image_path=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.", result=None, image_path=None)

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.", result=None, image_path=None)

    try:
        # Load image
        img = Image.open(io.BytesIO(file.read()))

        # Save image for display
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        img.save(image_path)

        # Get Prediction
        predicted_class, _, _ = predict_class(img)

        # Generate response
        result = {
            "diagnosis": predicted_class,
            "message": f"✅ **Diagnosed as:** {predicted_class}" if predicted_class.lower() == "normal" 
            else f"⚠️ **Detected:** {predicted_class}. Consult a doctor.",
            "warning": "⚠️ Cysts may indicate an underlying tumor. Further evaluation is recommended."
            if predicted_class.lower() == "cyst" else None
        }

        return render_template('index.html', result=result, image_path=image_path)

    except Exception as e:
        return render_template('index.html', error=str(e), result=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
