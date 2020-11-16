import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

import tensorflow as tf

from tqdm import tqdm
from glob import glob

from keras.models import load_model
from keras.preprocessing import image 
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

from extract_bottleneck_features import *

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '../images'
#UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}


#%matplotlib inline

#  Loading Data and Models
    
#  load list of dog names

app = Flask(__name__)  

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.config['SESSION_TYPE'] = 'filesystem'


def load_data_models():
    global dog_names
    global face_cascade, ResNet50_model, dog_breed_model
    global graph
    
    graph = tf.get_default_graph()

    print("\nLoading List of Dog Names")
    #dog_names = [item[20:-1].split('.')[1].replace('-', ' ').replace('_', ' ').title() for item in sorted(glob("../dogImages/train/*/"))]
    with open("./static/dog_names.txt", "rb") as fp:   # Unpickling
        dog_names = pickle.load(fp)

    print("*** Done ***")

    start_time = time.time()
    print('\nTo run the module efficiently the use of a GPU is recommended.\nImporting predictors without GPU support will extend import time signficiantly')

    #  load pre-trained face detector
    print("\nLoading Human Face Detector")
    face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')
    
    face_time = time.time()
    print("*** Loading face detector took {:.2f} seconds ***".format(time.time() - start_time))

    # load pre-trained dog image predictor model
    print("\nLoading Dog Detector")
    print("This can take several minutes. Patience")
    ResNet50_model = ResNet50(weights='imagenet')
    
    dog_time = time.time()
    print("*** Loading dog predictor model took {:.2f} seconds ***".format(dog_time - face_time))
    print("*** So far it took {:.2f} seconds ***".format(dog_time - start_time))
    
    # load pre-trained dog breed predictor model
    print("\nLoading Dog Breed Predictor")
    print("This can take several minutes. Patience\n")
    dog_breed_model = load_model('../saved_models/Xception_model.h5')
    
    print("*** Loading dog breed predictor took {:.2f} seconds ***".format(time.time() - dog_time))
    print("\n*** Overall loading took {:.2f} seconds ***".format(time.time() - start_time))

    return

def path_to_tensor(img_path):
    '''
    This function preprocesses the image and converts it from a 3 dimensional tensor
    to a 4 dimensional tensor compatible with Keras
    
    Input: path to an image
    
    Output: 4D tensor representing image and compatible with Keras
    '''
    
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def face_detector(img_path):
    '''
    This function takes an image and detects if the image represents a human face or not
    
    Input: image file path
    
    Output: True if image represents a human face
            False if image does not represent a human face
    '''
    try:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        print("Face Detected? {}".format(len(faces)))
    except:
        return len('')

    return len(faces) > 0

def ResNet50_predict_labels(img_path):
    '''
    returns prediction vector for image located at img_path
    
    Input:  image path
    
    Output: prediction vector
    '''
    
    ResNet50_model = ResNet50(weights='imagenet')

    img = preprocess_input(path_to_tensor(img_path))
    label = np.argmax(ResNet50_model.predict(img))
    #label = 268
    return label

def dog_detector(img_path):
    '''
    returns "True" if a dog is detected in the image stored at img_path
    
    Input:  image path
    
    Output: True if image represents a dog
            False if image does not represent a dog
    '''
    
    prediction = ResNet50_predict_labels(img_path)
    print("Dog Prediction: {}".format(prediction))

    return ((prediction <= 268) & (prediction >= 151)) 

def my_Xc_predict_breed(img_path):
    """
    This function predicts the dog breed fomr an image
    
    Input:  image path
    
    Output:  predicted dog breed
    """
    
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    
    # obtain predicted vector
    predicted_vector = dog_breed_model.predict(bottleneck_feature)
    
    # get index with highest probability and predicted dog breed
    predicted_index = np.argmax(predicted_vector)
    predicted_breed = dog_names[predicted_index]
    
    # replace separators '-' and '_' with spaces and capitalize first letters of breed name
    predicted_breed = predicted_breed.replace('-', ' ').replace('_', ' ').title()
    
    return predicted_breed

def show_image(img_path):
    # PLot the image
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show()


def image_predictor(img_path):
    
    """
    Function accepts an image path, attempts to determine if human face
    or a dog is present, and returns the predicted breed or breed
    the human face resembles.
    
    Input:  path to an image
    
    Output:  - if dog: image of dog with caption including predicted dog breed
             - if human: image of human with caption including predicted dog breed human looks like
             - if neither dog or human is detected :  image with caption that it is not from this world
    """

    print("\n{}\n".format(img_path))

    # check for each type and get prediction
    
    human = face_detector(img_path)
    dog = dog_detector(img_path)
    predicted_breed = my_Xc_predict_breed(img_path)
    
    # PLot the image
    #img = cv2.imread(img_path)
    #rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(rgb)
    #plt.show()
    
    # Caption the image
    if dog:
        breed = "What was the {} looking for?".format(predicted_breed)
        #print("What is this {} looking for?".format(predicted_breed))
    elif human:
        breed = "There was a resemblance with a {}.".format(predicted_breed)
        #print("You might resemble your pet {}.".format(predicted_breed))
    else:
        breed = "It most definitely was an Alien Creature from out of this world."
        #print("You most definitely are an Alien from out of this world.")
    
    return breed

def allowed_file(filename):
    '''
    Check if filename has an allowed extension
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def rename_file(filename):
    '''
    rename a file from name.ext to image.ext
    '''

    new_name = 'image.' + filename.rsplit('.', 1)[1].lower()

    return new_name

@app.route('/',methods=['GET','POST'])


def master():
    pred_class = ''
    path2img = 'https://i.ibb.co/ZVFsg37/default.png'

    if request.method == 'POST':
        # save selected image file in file
        file = request.files['img_sel']

        # check if the post request has the file part
        if file.filename == '':
            flash('No selected file')
            pred_class = ''
            return redirect(request.url)

        if 'img_sel' not in request.files:
            print('No file part')
            return redirect(request.url)

        filename = secure_filename(file.filename)

        # saving selected image file as image.ext in the images folder
        path2img = os.path.join(app.config['UPLOAD_FOLDER'], rename_file(filename))
        file.save(path2img)

        # predicting dog breed
        with graph.as_default():
            pred_class = image_predictor(path2img)
        
        return render_template('master.html',name=pred_class, imgpath=path2img)
    return render_template('master.html')

    

def main():
    load_data_models()
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()