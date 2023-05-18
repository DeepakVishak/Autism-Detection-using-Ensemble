# from model import model
import os
from keras_preprocessing.image import ImageDataGenerator, load_img
import cv2
from tensorflow.keras.utils import img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np


def preprocess_for_model1(img):
    
    timg = load_img(img, color_mode='grayscale', target_size=(224, 224))

    # Convert image to numpy array and apply Gaussian blur
    timg = img_to_array(timg)
    timg = cv2.GaussianBlur(timg, (5, 5), 0)

    # Convert image to 8-bit unsigned integer
    timg = cv2.convertScaleAbs(timg)

    # Apply adaptive thresholding and Otsu's thresholding
    th3 = cv2.adaptiveThreshold(timg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, test_image = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Reshape the image to (1, 224, 224, 1) and convert back to float32
    test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
    test_image = test_image.reshape((1, 224, 224, 3)).astype(np.float32)

    return test_image


def preprocess_for_model2(img):
    
    test_image = np.array(img.resize((224, 224)))
    return np.expand_dims(test_image, axis=0)

def prediction(model_1,model_2,test_img_model1,test_img_model2):

    class_names = ['Autistic', 'Non Autistic']
    predictions_1 = model_1.predict(test_img_model1)
    predictions_2 = model_2.predict(test_img_model2)


    scores = []
    for pred_1, pred_2 in zip(predictions_1, predictions_2):
        score_1 = tf.nn.softmax(pred_1)
        score_2 = tf.nn.softmax(pred_2)
    
        # If both models predict the same class
        if np.argmax(score_1) == np.argmax(score_2):
            class_index = np.argmax(score_1)
            avg_score = (np.max(score_1) + np.max(score_2)) / 2
            scores.append((class_index, avg_score))
    
        # If both models predict different classes
        else:
            max_score_1 = np.max(score_1)
            max_score_2 = np.max(score_2)
            if max_score_1 > max_score_2:
                scores.append((np.argmax(score_1), max_score_1))
            else:
                scores.append((np.argmax(score_2), max_score_2))
    
    if len(scores) == 1:
        result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[scores[0][0]], 100 * scores[0][1])
    elif scores[0][0] == scores[1][0]:
        avg_score = (scores[0][1] + scores[1][1]) / 2
        result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[scores[0][0]], 100 * avg_score)
    else:
        result = "Cannot predict with 50% confidence."
    
    return result



def predict():
    
    #Load two models
    model_1=load_model('customized_model.h5')
    model_2=load_model('customized_model_without_preprocess.h5')

    matplotlib.use('agg')
    test_img = r"static/files/image.jpg"
    # Open the image from the file path
    image = Image.open(test_img)

    # Display the image using matplotlib
    plt.imshow(image)
    plt.show()

    test_img_model1 = preprocess_for_model1(test_img)
    test_img_model2 = preprocess_for_model2(image)

    x = prediction(model_1,model_2,test_img_model1,test_img_model2)
    return x

