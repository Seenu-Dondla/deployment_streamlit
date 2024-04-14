from flask import Flask, request, jsonify ,render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)
model = load_model('EffNet-model.h5')

def preprocess_image(file_storage, target_size=(224, 224)):
    file_storage.seek(0)  # Reset file pointer to the beginning
    img = image.load_img(io.BytesIO(file_storage.read()), target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    img_array = preprocess_image(file)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)

    if predicted_class_index == 0:
        result = {'result': '\t CONGRATULATIONS\n\n\n !!! You are Healthy !!!'}
    elif predicted_class_index == 1:
        result = {'result': 'You are affected by Glaucoma \n\n Glaucoma Level : MILD \n\n Remedy : \n \u2022 Regular Eye Exams \n\u2022 Wear protective eyewear \n\u2022 Maintain a healthy lifestyle'}
    elif predicted_class_index == 2:
        result = {'result': 'You are affected by Glaucoma \n\n Glaucoma Level : MODERATE \n\n Remedy : \n \u2022 Regular Eye Exams \n \u2022 Manage diabetes and hypertension \n \u2022 Maintain a healthy lifestyle'}
    elif predicted_class_index == 3:
        result = {'result': 'You are affected by Glaucoma \n\n Glaucoma Level : PROLIFERATE(HIGH) \n\n Remedy : \n \u2022 Combination Therapies \n \u2022 Quit bad habits \n \u2022 Daily Meditation \n\u2022 Maintain a healthy lifestyle '}
    elif predicted_class_index == 4:
        result = {'result': 'You are affected by Glaucoma \n\n Glaucoma Level : SEVERE \n\n Remedy : \n \u2022 Laser Therapy \n \u2022 Medications \n \u2022 Daily Meditation \n \u2022 Maintain a healthy lifestyle '}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
