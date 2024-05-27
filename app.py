from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# age_predictor_model = load_model('models/age_predictor.h5')
# breed_classifier_model = load_model('models/breed_classifier.h5')

def cut_dog(image):
    # to be replaced with yolo model
    width, height = image.size
    left = (width - 224) // 2
    top = (height - 224) // 2
    right = left + 224
    bottom = top + 224
    dog_image = image.crop((left, top, right, bottom))
    return dog_image

def predict_age(image):
    # to be replaced with actual model
    # age = age_predictor_model.predict(processed_img)
    age = np.random.randint(0, 3)
    return age

def predict_breed(image):
    # to be replaced with actual model
    # breed = breed_classifier_model.predict(processed_img)
    breed = "golden_retriever"
    return breed

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'file error'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'file error'}), 400

    img = Image.open(io.BytesIO(file.read()))
    dog_img = cut_dog(img)
    age = predict_age(dog_img)
    breed = predict_breed(dog_img)

    response = {
        'age': age,
        'breed': breed
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
