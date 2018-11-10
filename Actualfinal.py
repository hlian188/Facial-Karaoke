
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

#Now run it





    # initialize the camera
    #cam = cv2.VideoCapture(0) # 0 -> index of camera
    
    
        
###

from tkinter import *
import random 
import pygame

####################################
# init
####################################


def init(data):
    # There is only one init, not one-per-mode
    run_training()
    data.cam = cv2.VideoCapture(0)
    data.mode = "splashScreen"

    data.fontSize = [26, 36]
    data.timerCalls = 0
    data.index = 1
    data.message = "welcome to a blast from the past"    

    data.emojis = [ ([0] * 3) for row in range(6) ]

    data.emojis[0][0] = PhotoImage(file="happy.gif")
    data.emojis[1][0] = PhotoImage(file="disgust.gif")
    data.emojis[2][0] = PhotoImage(file="angry.gif")
    data.emojis[3][0] = PhotoImage(file="sad.gif")
    data.emojis[4][0] = PhotoImage(file="neutral.gif")
    data.emojis[5][0] = PhotoImage(file="pikachu.gif")
    data.emoji = random.randint(0,5)

    data.iconSize = 200
    count = 0

    for i in range(len(data.emojis)):
        data.emojis[i][1] = count      #x-coordinate
        data.emojis[i][2] = 0          #y-coordinate
        count += 200
        
    data.score = 0
    
####################################
# mode dispatcher
####################################

def mousePressed(event, data):
    if (data.mode == "splashScreen"): splashScreenMousePressed(event, data)
    elif (data.mode == "playGame"):   playGameMousePressed(event, data)
    elif (data.mode == "instruct"):    instructMousePressed(event, data)
    elif (data.mode == "help"):    helpMousePressed(event, data)

def keyPressed(event, data):
    if (data.mode == "splashScreen"): splashScreenKeyPressed(event, data)
    elif (data.mode == "playGame"):   playGameKeyPressed(event, data)
    elif (data.mode == "instruct"):   instructKeyPressed(event, data)
    elif (data.mode == "help"):   helpKeyPressed(event, data)

def timerFired(data):
    if (data.mode == "splashScreen"): splashScreenTimerFired(data)
    elif (data.mode == "playGame"):   playGameTimerFired(data)
    elif (data.mode == "instruct"):   instructTimerFired(data)
    elif (data.mode == "help"):   helpTimerFired(data)

def redrawAll(canvas, data):
    if (data.mode == "splashScreen"): splashScreenRedrawAll(canvas, data)
    elif (data.mode == "playGame"):   playGameRedrawAll(canvas, data)
    elif (data.mode == "instruct"):   instructRedrawAll(canvas, data)
    elif (data.mode == "help"):   helpRedrawAll(canvas, data)

####################################
# splashScreen mode
####################################

def splashScreenMousePressed(event, data):
    pass

def splashScreenKeyPressed(event, data):
    if event.keysym == "space":
        data.mode = "instruct"

def splashScreenTimerFired(data):
    data.timerCalls += 1
    if data.timerCalls % 5 == 0:
        data.index += 1

def splashScreenRedrawAll(canvas, data):
    canvas.create_rectangle(0, 0, data.width, data.height, fill="black")
    canvas.create_text(data.width/2, data.height-390, text="MIMIC ME", 
                       font="Arial, 72 bold", fill="white")
    canvas.create_text(data.width/2, data.height-310,
                       text="Press the space bar to play!", font="Arial 20",
                       fill="white")
    canvas.create_text(data.width/2, data.height-270,
                       text="Press 'p' to pause in the game!", font="Arial 20",
                       fill="white")

    #colored welcome message                   
    cycle = -1
    gap = 18
    colors = ["red3", "firebrick1", "DarkOrange1", "chocolate1", "orange1", 
            "orange", "gold", "yellow2", "yellow", "OliveDrab1", "SpringGreen2",
            "lawn green", "chartreuse2", "green2", "green3", "CadetBlue2",
            "turquoise1", "cyan", "cyan2", "SteelBlue1", "DeepSkyBlue2",
            "RoyalBlue1", "RoyalBlue3", "SlateBlue", "purple1", "purple3", 
            "dark violet", "medium orchid", "violet red", "DeepPink2", 
            "maroon1", "VioletRed1"]

    colorLength = len(colors)

    for c in data.message:
        cycle += 1
        x0 = 124 + (cycle*gap)
        y0 = data.height-150
        x1 = 124 + (cycle*gap) + gap
        y1 = data.height-150
        canvas.create_text((x0+x1/2), y0, text=c, 
                    fill=colors[cycle%colorLength], 
                    font="Arial "+str(data.fontSize[data.index%2])+" bold")

####################################
# instruct mode
####################################

def instructMousePressed(event, data):
    pass

def instructKeyPressed(event, data):
    if event.keysym == "space":
        pygame.mixer.init()
        pygame.mixer.music.load("guacamole.mp3")
        pygame.mixer.music.play()
        data.mode = "playGame"

def instructTimerFired(data):
    pass

def instructRedrawAll(canvas, data):
    canvas.create_rectangle(0, 0, data.width, data.height, fill="black")
    canvas.create_text(data.width/2, data.height/4,
                        text="Match your face to the emoji to score points!",
                        font="Arial 42 bold", fill="white")

    #images
    canvas.create_image(data.iconSize/2, data.height/2, image=data.emojis[0][0])
    canvas.create_image(data.iconSize + (data.iconSize/2), data.height/2,
                        image=data.emojis[1][0])
    canvas.create_image((2*data.iconSize) + data.iconSize/2, data.height/2,
                        image=data.emojis[2][0])
    canvas.create_image((3*data.iconSize) + data.iconSize/2, data.height/2,
                        image=data.emojis[3][0])
    canvas.create_image((4*data.iconSize) + data.iconSize/2, data.height/2,
                        image=data.emojis[4][0])
    canvas.create_image((5*data.iconSize) + data.iconSize/2, data.height/2,
                        image=data.emojis[5][0])

    #text below image
    canvas.create_text(data.iconSize/2, data.height-200, text="Happy", 
                        fill="white", font="Arial 26 bold")
    canvas.create_text(data.iconSize + (data.iconSize/2), data.height-200,
                        text="Disgust", fill="white", font="Arial 26 bold")
    canvas.create_text((2*data.iconSize) + (data.iconSize/2), data.height-200,
                        text="Angry", fill="white", font="Arial 26 bold") 
    canvas.create_text((3*data.iconSize) + (data.iconSize/2), data.height-200,
                        text="Sad", fill="white", font="Arial 26 bold")
    canvas.create_text((4*data.iconSize) + (data.iconSize/2), data.height-200,
                        text="Neutral", fill="white", font="Arial 26 bold")
    canvas.create_text((5*data.iconSize) + (data.iconSize/2), data.height-200,
                        text="Shocked", fill="white", font="Arial 26 bold")

    canvas.create_text(data.width/2, data.height-120, 
            text="Let the games begin :\")", fill="white", font="Arial 30 bold")
    canvas.create_text(data.width/2, data.height-60, 
            text="Press the space bar to continue", fill="white", 
            font="Arial 26")

####################################
# help mode
####################################

def helpMousePressed(event, data):
    pass

def helpKeyPressed(event, data):
    if event.keysym == "space":
        data.mode = "playGame"

def helpTimerFired(data):
    pass

def helpRedrawAll(canvas, data):
    canvas.create_rectangle(0, 0, data.width, data.height, fill="black")
    canvas.create_text(data.width/2, data.height/4,
                        text="Match your face to the emoji to score points!",
                        font="Arial 42 bold", fill="white")

    #images
    canvas.create_image(data.iconSize/2, data.height/2, image=data.emojis[0][0])
    canvas.create_image(data.iconSize + (data.iconSize/2), data.height/2,
                        image=data.emojis[1][0])
    canvas.create_image((2*data.iconSize) + data.iconSize/2, data.height/2,
                        image=data.emojis[2][0])
    canvas.create_image((3*data.iconSize) + data.iconSize/2, data.height/2,
                        image=data.emojis[3][0])
    canvas.create_image((4*data.iconSize) + data.iconSize/2, data.height/2,
                        image=data.emojis[4][0])
    canvas.create_image((5*data.iconSize) + data.iconSize/2, data.height/2,
                        image=data.emojis[5][0])

    #text below image
    canvas.create_text(data.iconSize/2, data.height-200, text="Happy", 
                        fill="white", font="Arial 26 bold")
    canvas.create_text(data.iconSize + (data.iconSize/2), data.height-200,
                        text="Disgust", fill="white", font="Arial 26 bold")
    canvas.create_text((2*data.iconSize) + (data.iconSize/2), data.height-200,
                        text="Angry", fill="white", font="Arial 26 bold") 
    canvas.create_text((3*data.iconSize) + (data.iconSize/2), data.height-200,
                        text="Sad", fill="white", font="Arial 26 bold")
    canvas.create_text((4*data.iconSize) + (data.iconSize/2), data.height-200,
                        text="Neutral", fill="white", font="Arial 26 bold")
    canvas.create_text((5*data.iconSize) + (data.iconSize/2), data.height-200,
                        text="Shocked", fill="white", font="Arial 26 bold")

    canvas.create_text(data.width/2, data.height-100, 
            text="Press the space bar to continue", fill="white", 
            font="Arial 26")

####################################
# playGame mode
####################################

def playGameMousePressed(event, data):
    data.score = 0

def playGameKeyPressed(event, data):
    if (event.keysym == 'p'):
        data.mode = "help"
        
def playGameTimerFired(data):
    s, img = data.cam.read()
    if s:    # frame captured without any errors
    
    #cv2.namedWindow("cam-test",CV_WINDOW_AUTOSIZE)
        cv2.imshow("cam-test",img)
        cv2.waitKey(50)
        #cv2.destroyWindow("cam-test")
        cv2.imwrite("filename.jpg",img)
        readInEmotion = run_recognizer("filename.jpg")
        
    
    
    data.emojis[data.emoji][2] += 15

    if data.emojis[data.emoji][2] >= data.height - data.iconSize:
        if readInEmotion == 0 and data.emoji == 4:
            data.score += 1
        elif readInEmotion == 1 and data.emoji == 2:
            data.score += 1
        elif readInEmotion == 3 and data.emoji == 1:
            data.score += 1
        elif readInEmotion == 5 and data.emoji == 0:
            data.score += 1
        elif readInEmotion == 6 and data.emoji == 3:
            data.score += 1
        elif readInEmotion == 7 and data.emoji == 5:
            data.score += 1

        data.emoji = random.randint(0,5)
        data.emojis[data.emoji][2] = 0 
        data.score += 0

def playGameRedrawAll(canvas, data):
    for i in range (len(data.emojis)):
        canvas.create_image(data.emojis[i][1], data.height, anchor = SW, 
                            image = data.emojis[i][0])
                            
    canvas.create_image(data.emojis[data.emoji][1], data.emojis[data.emoji][2],
                        anchor = NW, image = data.emojis[data.emoji][0])
    canvas.create_text(data.width//2, 0, text = "Score: " + str(data.score), 
                        font = "Arial 20", fill = "black", anchor = N)

####################################
# use the run function as-is
####################################

def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()    

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)

    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 50 # milliseconds
    root = Tk()
    root.resizable(width=False, height=False) # prevents resizing window
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")


run(1200, 700)
        
        
