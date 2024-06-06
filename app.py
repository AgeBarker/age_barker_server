from flask import Flask, request, jsonify
from flask_cors import CORS
# from keras.models import load_model
from PIL import Image
import numpy as np
import io, dlib, cv2
import tensorflow as tf

app = Flask(__name__)
CORS(app)

print("Loading models...")
print("Loading breed_classifier_model...")
breed_classifier_model = tf.keras.models.load_model('models/breed_classifier.h5')
print("Loading age_predictor_model...")
age_predictor_model = tf.keras.models.load_model('models/age_predictor.h5')
print("Models loaded!")

IMAGE_SIZE = (331, 331)
IMAGE_FULL_SIZE = (331, 331, 3)     # z jakiegoś powodu wytrenowałam sieć dla obrazków 331x331 

classes_age = {
    0: 'young',
    1: 'adult',
    2: 'old'
}

classes = np.load("models/train_labels.npy")
classes = classes.tolist()
classes_int = np.load("models/labels_enc.npy")
mapping = dict(zip(classes_int, classes))


def cut_dog(image):
    print("DUUUUUUUPA twarze psa")
    detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')

    print("DUUUUUUUP 2")
    i = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(i, (640, 480))
    dets = detector(img, upsample_num_times=1)

    print("DUUUUUUUP 3")
    if len(dets) == 0:
        return None
    x1, y1 = dets[0].rect.left(), dets[0].rect.top()
    x2, y2 = dets[0].rect.right(), dets[0].rect.bottom()
    print("DUUUUUUUP 4")
    dog_image = img[y1:y2, x1:x2]
    cv2.imwrite('dog.jpg', dog_image)
    return dog_image

def preprocess(image, img_size):
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA) / 255.0
    image = tf.expand_dims(image, 0)
    return image
    

def predict_age(image):
    preprocess_image = preprocess(image, (224, 224))
    age = age_predictor_model.predict(preprocess_image)
    age = np.argmax(age)
    # age = np.random.randint(0, 3)
    return age

def predict_breed(image):
    preprocess_image = preprocess(image, (331, 331))
    breed = breed_classifier_model.predict(preprocess_image)

    breed = mapping[np.argmax(breed)]
    # breed = "golden_retriever"
    return breed

@app.route('/')
def index():
    return jsonify({
        'age': classes_age[0],
        'breed': 'golden_retriever'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'file error'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'file error'}), 400

    # img = Image.open(io.BytesIO(file.read()))
    # print(img)
    pil_image = Image.open(io.BytesIO(file.read())).convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    dog_img = cut_dog(open_cv_image)
    age = predict_age(dog_img)          # 0 - young, 1 - adult, 2 - old zamienić na string
    breed = predict_breed(dog_img)      # nazwa rasy psa

    response = jsonify({
        'age': classes_age[age],
        'breed': breed
    })

    print(response.data)

    return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
