from flask import Flask  , render_template, request ,jsonify
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

detector= MTCNN()
model= VGGFace(model='resnet50' , include_top=False ,input_shape=(224,224,3), pooling='avg')

feature_list = np.array(pickle.load(open('embedding.pkl','rb' )))
filenames= pickle.load(open('filenames.pkl','rb' ))


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False


def extract_features(img_path,model,detector):
    img= cv2.imread(img_path)
    results = detector.detect_faces(img)   
    x, y, width, height = results[0]['box']
    face= img[y:y+height,x:x+width]

    image= Image.fromarray(face)
    image= image.resize((224,224))

    face_array= np.asarray(image)

    face_array=face_array.astype('float32')


    expanded =np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded)

    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features):
    similarity=[]
    for i in range(len(feature_list)):
   
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0, 0])

    index_pos= sorted(list(enumerate(similarity)),reverse =True,key=lambda x: x[1] )[0][0]
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
    
    if file:
        # Save uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
    features=extract_features(os.path.join('uploads',file.filename),model,detector)
    print(features.shape)
    index_pos=recommend(feature_list, features)
    print(index_pos)
    return filenames[index_pos]


if __name__ == '__main__':
    app.run(debug=True)