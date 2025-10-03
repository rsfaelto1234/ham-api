from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Ruta del modelo
MODEL_PATH = "modelo_cnn_ham10000.h5"

# Cargar el modelo CNN
model = load_model(MODEL_PATH)

# Clases entrenadas (ajusta si difieren)
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
mel_index = class_names.index('mel')  # índice de melanoma

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API CNN HAM10000 funcionando correctamente 🚀"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Debes enviar una imagen con la clave 'file'"}), 400

    file = request.files['file']
    file_path = "temp.jpg"
    file.save(file_path)

    # Preprocesar la imagen
    img = image.load_img(file_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Predicción
    probs = model.predict(x)[0]
    melanoma_prob = probs[mel_index] * 100

    # Clasificar según rango
    if melanoma_prob <= 45.99:
        estado = "NEGATIVO"
        mensaje = "LA IMAGEN FUE EVALUADA CORRECTAMENTE (NEGATIVO)"
    else:
        estado = "POSITIVO"
        mensaje = "LA IMAGEN FUE EVALUADA CORRECTAMENTE COMO (POSITIVO) ACUDIR A UN CENTRO ESPECIALIZADO"

    # Respuesta JSON
    return jsonify({
        "probabilidad": round(float(melanoma_prob), 2),
        "estado": estado,
        "mensaje": mensaje
    })

if __name__ == '__main__':
    # Render asigna automáticamente el puerto
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
