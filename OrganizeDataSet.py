import glob
from shutil import copyfile
import os
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
participants = glob.glob("source_emotion\\*") #Returns a list of all folders with participant numbers
for x in participants:
    part = "%s" %x[-4:] #store current participant number
    for sessions in glob.glob("%s\\*" %x): #Store list of sessions for current participant
        for files in glob.glob("%s\\*" %sessions):
            print(files)
            
            current_session = files[20:-30]
            print("sessions:", sessions)
            print(files)
            
            file = open(files, 'r')
            x = file.readline()[3:10]
            print(type(x))
            x = float(x)
            print(x)
            print(type(x))
            print("part and curr sess:", part, current_session)
            emotion = int(float(x))#emotions are encoded as a float, readline as float, then convert to integer.
            sourcefile_emotion= glob.glob("source_images\\%s\\%s\\*" %(part, current_session))
            print(sourcefile_emotion) #should be a list
            sourcefile_emotion = glob.glob("source_images\\%s\\%s\\*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
            sourcefile_neutral = glob.glob("source_images\\%s\\%s\\*" %(part, current_session))[0] #do same for neutral image
            dest_neut = "sorted_set\\neutral\\%s" %sourcefile_neutral[25:] #Generate path to put neutral image
            dest_emot = "sorted_set\\%s\\%s" %(emotions[emotion], sourcefile_emotion[25:]) #Do same for emotion containing image
            copyfile(sourcefile_neutral, dest_neut) #Copy file
            copyfile(sourcefile_emotion, dest_emot) #Copy file