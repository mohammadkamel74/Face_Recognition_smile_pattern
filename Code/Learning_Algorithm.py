################################################import libraries################
import csv
import warnings
import dlib
import math
import cv2
import os
from imutils import face_utils
import numpy as np
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
import matplotlib.pyplot as plt
################################################define variables################
counter=0
ArrayX = np.empty((13,), dtype=object)
ArrayY = np.empty((13,), dtype=object)
XX=[]
YY=[]
Coefficients=[]
NumberHappyFace=0
NumberTotallFace=0
#################################################different emotional status#####
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#################################################CNN Model for emotion detection
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
######################################################Load model################
model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)
################################################define Haar Cascade and dlib####
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
################################################identify video##################
DIR = '../Training_Data'
CATEGORIES=[]
directory_contents = os.listdir(DIR)
for item in directory_contents:
    if not os.path.isdir(item):
        CATEGORIES.append(item)
print(CATEGORIES)
#DIR = 'Data'
#CATEGORIES = ["Azin"]
#CATEGORIES = ["Azin","Hadi","Mahsa","Mohammad","Nadia","Parisa", "Rojan","Shaghayegh","Sister1","Soheil","Zahra"]
for CATEGORY in CATEGORIES:
    path = os.path.join(DIR, CATEGORY)  # paths to the legobricks
    class_num = CATEGORIES.index(CATEGORY)
    for Video in os.listdir(path):
        counter=0
        NumberHappyFace=0
        NumberTotallFace=0
        cap = cv2.VideoCapture(os.path.join(path, Video))
        ##########################################ShiTomasi corner detection####
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        ##########################################lucas kanade optical flow ####
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(40, 40),
                         maxLevel=5,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.001))
        ##########################################frame preprocessing###########
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        height1, width1 = old_frame.shape[:2]
        ##########################################find face in frame## ###############################################
        faces = face_cascade.detectMultiScale(old_gray, 1.1, 1, minSize=(40, 40))
        ##########################################frame preprocessing###############################################
        for (x, y, w, h) in faces:
            print("ok")
        old_crop_img = old_frame[y-20:y+h+20, x-20:x+w+20]
        height, width = old_frame.shape[:2]
        old_crop_img1=old_crop_img.copy()
        old_crop_img = cv2.cvtColor(old_crop_img, cv2.COLOR_BGR2GRAY)
        ##########################################fine landmarks by dlib###############################################
        faces = detector(old_crop_img)
        for face in faces:
            landmarks = predictor(old_crop_img, face)
            shape = face_utils.shape_to_np(landmarks)
            shape = np.array(shape).reshape(-1, 2)
            cheeksleft = (shape[4][0] + int((shape[48][0] - shape[4][0]) / 2),
                                           shape[29][1] + int((shape[33][1] - shape[29][1]) / 2))
            cheeksright = (shape[54][0] - int((shape[54][0] - shape[12][0]) / 2),
                                            shape[29][1] + int((shape[33][1] - shape[29][1]) / 2))
            ntotal = 13
            k = 0
            array = np.ndarray((ntotal, 1, 2))
            array[k]=cheeksleft
            k+=1
            array[k] = cheeksright
            k += 1
            array[k] = shape[31]
            k += 1
            array[k] = shape[35]
            k += 1
            array[k] = shape[36]
            k += 1
            array[k] = shape[45]
            k += 1
            array[k] = shape[48]
            k += 1
            array[k] = shape[49]
            k += 1
            array[k] = shape[51]
            k += 1
            array[k] = shape[53]
            k += 1
            array[k] = shape[54]
            k += 1
            array[k] = shape[56]
            k += 1
            array[k] = shape[58]
            k += 1
            pp=0
            pp = array.copy()
            pp = np.float32(pp)
            mask = np.zeros_like(old_crop_img1)
        ##########################################show landmarks on first  frame###########################################
        for i in range(len(pp)):
            XX.append(pp[i][0][0])
            YY.append(pp[i][0][1])
            cv2.circle(old_crop_img, (int(pp[i][0][0]), int(pp[i][0][1])), 4, (255, 0, 0), -1)
        ###############################################save points in array#################################################
        for i, v in enumerate(ArrayX):
            ArrayX[i] = [pp[i][0][0]]
        for i, v in enumerate(ArrayY):
            ArrayY[i] = [pp[i][0][1]]
        ######################################################start while######################################################
        while (1):
            counter = counter + 1
            if counter <= 200:
                ret, frame = cap.read()
                if ret==1:
                    ##########################################frame preprocessing###############################################
                    crop_img1 = frame[y-20:y+h+20, x-20:x+w+20]
                    height, width = frame.shape[:2]
                    crop_img = cv2.cvtColor(crop_img1, cv2.COLOR_BGR2GRAY)
                    #cv2.imshow('frame', crop_img) 
                    #cv2.waitKey(2) & 0xff         
                    ######################################find and print emotion###############################################
                    mcropped_img = np.expand_dims(np.expand_dims(cv2.resize(crop_img, (48, 48)), -1), 0)
                    mprediction = model.predict(mcropped_img)
                    mmaxindex = int(np.argmax(mprediction))
                    if emotion_dict[mmaxindex] == "Happy":
                        NumberHappyFace = NumberHappyFace + 1
                    NumberTotallFace = NumberTotallFace + 1
                    ##########################################Calculate optical flow############################################
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_crop_img, crop_img, pp, None, **lk_params)
                    #############################################Select good points#############################################
                    good_new = p1[st == 1]
                    good_old = pp[st == 1]
                    #############################################draw the tracks################################################
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                                a, b = new.ravel()
                                c, d = old.ravel()
                                mask = cv2.line(mask, (a, b), (c, d),(120, 20, 222), 2)
                                crop_img1 = cv2.circle(crop_img1, (a, b), 5, (76, 76, 45), -1)
                    img = cv2.add(crop_img1,mask)
                    #cv2.imshow('frame', img)   
                    #cv2.waitKey(100) & 0xff    
                    ###############################dNow update the previous frame and previous points############################
                    old_crop_img = crop_img.copy()
                    pp = good_new.reshape(-1, 1, 2)
                    ##########################################save points in array##############################################
                    if len(pp)<13:
                        continue
                    try:
                        for i, v in enumerate(ArrayX): v.append(pp[i][0][0])
                        for i, v in enumerate(ArrayY): v.append(pp[i][0][1])
                    except:
                        print("error")
            if counter > 200:
                break
        cv2.destroyAllWindows()
        cap.release()
        ArrayX=np.asarray(ArrayX)
        ArrayY=np.asarray(ArrayY)
        print("number of frames containing face with happy emotion: " + str(NumberHappyFace))
        print("number of frames: "+ str(NumberTotallFace))
        if int(0.2*NumberTotallFace)<=NumberHappyFace:
            ###############################################Curve fitting########
            Coefficients=[]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                for i, v in enumerate(ArrayX):
                    p30 = np.poly1d(np.polyfit(ArrayX[i], ArrayY[i], 4))
                    Coefficients.append(p30)
                    #print(p30.coef)
            #print(Coefficients)
        else :
            print("No happy Face")
        ################################################Save features in excel##
        with open('Train1.csv', 'a') as td:
            for i in range(13):
                co=list((Coefficients[i].coef))
                co.append(class_num)
                #co.append(genderval)
                td.write(str(co))
                td.write('\n')