import cv2
import glob
import random
import numpy as np
#from ExtractingFaces import *
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier
data = {}
def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    
    return training #list
def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training: #item is an image
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
        
    return training_data, training_labels

def run_training():
    training_data, training_labels = make_sets()
    print("training fisher face classifier")
    print("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
    print("predicting classification set")
    
def run_recognizer(image, x = None):
    image = cv2.imread(image)
    image = cv2.resize(image, (350,350))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return fishface.predict(gray)[0] 
"""
def run_recognizer(image, x = None):
    if x == None:
        
        training_data, training_labels = make_sets()
        print("training fisher face classifier")
        print("size of training set is:", len(training_labels), "images")
        fishface.train(training_data, np.asarray(training_labels))
        print("predicting classification set")
        x = 1
        return run_recognizer(image, x)
    
    else:
        image = cv2.imread(image)
        image = cv2.resize(image, (350,350))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(type(fishface.predict(gray)))
        print(fishface.predict(gray)[0])
        return fishface.predict(gray)[0] 
"""
 
#Now run it
run_training()
test = run_recognizer("filename.jpg")
print(test)





"""
import cv2
import glob
faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions




def detect_faces(emotion):
    files = glob.glob("sorted_set\\%s\\*" %emotion) #Get list of all images with emotion
    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""
        #Cut and save face
        print("f:", f)
        print(type(f))
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            #print("face found in file: %s") %f
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("dataset\\%s\\%s.jpg" %(emotion, filenumber), out) #Write image
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number
for emotion in emotions:
    detect_faces(emotion) #Call functiona
"""



