from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image
# load img -> face detection and extract its features
# find the cosine distance of current image with all the 8655 features
# recommend that image
feature_list = np.array(pickle.load(open('embedding.pkl','rb' )))
filenames= pickle.load(open('filenames.pkl','rb' ))


model= VGGFace(model='resnet50' , include_top=False ,input_shape=(224,224,3), pooling='avg')
detector = MTCNN(min_face_size=20)

sample_img= cv2.imread('sample\\yash_pfp.jpg')

if sample_img is None:
    print("Error: Image not found. Check the path.")
    exit()


print("Image shape:", sample_img.shape)  # Check image dimensions
# cv2.imshow('Sample Image', sample_img)   # Display the image
# cv2.waitKey(0)                           # Wait for a key press to close the window
# cv2.destroyAllWindows()
results= detector.detect_faces(sample_img)
print("Detection Results:", results)  # Print what MTCNN detects

# Ensure a face is detected
if len(results) == 0:
    print("No face detected. Try another image.")
    exit()


x, y, width, height = results[0]['box']
face= sample_img[y:y+height,x:x+width]
# cv2.imshow('output', face)
# cv2.waitKey(0)

image= Image.fromarray(face)
image= image.resize((224,224))

face_array= np.asarray(image)

face_array=face_array.astype('float32')


expanded =np.expand_dims(face_array, axis=0)
preprocessed_img = preprocess_input(expanded)

result = model.predict(preprocessed_img).flatten()
similarity=[]
for i in range(len(feature_list)):
    # similarity.append(cosine_similarity(result.reshape(1,-1), feature_list[i].reshape(1,-1)[0][0]))
    similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0, 0])


index_pos= sorted(list(enumerate(similarity)),reverse =True,key=lambda x: x[1] )[0][0]


temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output', temp_img)
cv2.waitKey(0)