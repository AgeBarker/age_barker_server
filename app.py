from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import tensorflow as tf


app = Flask(__name__)

# age_predictor_model = load_model('models/age_predictor.keras')
# breed_classifier_model = load_model('models/breed_classifier.keras')

IMAGE_SIZE = (331, 331)
IMAGE_FULL_SIZE = (331, 331, 3)     # z jakiegoś powodu wytrenowałam sieć dla obrazków 331x331 

classes_age = {
    0: 'young',
    1: 'adult',
    2: 'old'
}

def cut_dog(image):
    detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(img, upsample_num_times=1)
    if len(dets) == 0:
        return None
    x1, y1 = dets[0].rect.left(), dets[0].rect.top()
    x2, y2 = dets[0].rect.right(), dets[0].rect.bottom()
    dog_image = img[y1:y2, x1:x2]
    dog_image = cv2.resize(dog_image, dsize=IMAGE_SIZE)
    return dog_image

def preprocess(image, img_size):
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
    image = tf.expand_dims(image, 0)
    image = image / 255.0
    return image
    

def predict_age(image):
    preprocess_image = preprocess(image, (224, 224))
    # to be replaced with actual model
    # processed_img = preprocess_image(image)
    # age = age_predictor_model.predict(processed_img)
    age = np.random.randint(0, 3)
    return age

def predict_breed(image):
    preprocess_image = preprocess(image, (331, 331))
    # to be replaced with actual model
    # processed_img = preprocess_image(image)
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
    age = predict_age(dog_img)          # 0 - young, 1 - adult, 2 - old zamienić na string
    breed = predict_breed(dog_img)      # nazwa rasy psa

    response = {
        'age': age,
        'breed': breed
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
