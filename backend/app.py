from flask import Flask, render_template, request, jsonify
import os
import cv2
from PIL import Image
from mtcnn import MTCNN
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Initialize MTCNN and VGGFace model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load precomputed embeddings and filenames
feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if len(results) == 0:
        return None

    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]

    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')

    expanded = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded)

    return model.predict(preprocessed_img).flatten()

def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0, 0] for f in feature_list]
    index_pos = np.argmax(similarity)
    return index_pos

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    features = extract_features(filepath, model, detector)
    if features is None:
        return jsonify({"error": "No face detected. Try another image."}), 400

    index_pos = recommend(feature_list, features)
    celeb_path = filenames[index_pos]

    celeb_name = os.path.splitext(os.path.basename(celeb_path))[0]
    print(celeb_path)
    return jsonify({
        "name": celeb_name,
        "image_url": f"{celeb_path}",
        "description": "You and this Bollywood star could be twins!"
    })

if __name__ == '__main__':
    app.run(debug=True)
