# Face_Recognition_Smile_Pattern

Face recognition is the problem of identifying and verifying people in a photograph by their face. Since every Smile is [Unique](https://arxiv.org/abs/1802.01873), in this project an attemp is made to propose Smile Pattern as a new biometric trait for face recognitionuse task. In this project we extract dlib facial landmarks and track their movements during video frames using optical flow. Finally, by using curve fitting we extract the coefficients of each landmark. Project steps are as follows:

### a) Video Stabilization

Since in our methodologies we are using optical flow, we need stable videos. Since most of the face recognition dataset videos are taken in undercontrolled setting, we must use stabilization approaches to remove unwanted noises and movments to have an smooth video. 

### b) Face Detection using Har Cascade

Haar Cascade is a machine learning object detection algorithm proposed by Paul Viola and Michael Jones. In this project  we used Haar Cascade method to extract the face in the first frame of video. In order to detect smile in video frames, we used CNN based facial expression tools.

### c) Smile Detection Using CNN

Since the video might not include a smile, we need to check whether the person smile or not. Therefore, we have used a pretrained CNN based model to count the number of frames containing smile. We calculate the facial expression for each frame and count the number of frames with happy tags. If the number of happy frames is greater than 0.2 of all frames, we conclude that the person has smile during the video and the smile pattern could be extracted.

### d) Facial Feature Extraction

Dlib library could extract 68 facial landmarks in an image. The landmarks that we are interested in, are the one that describes the shape of the face attributes like: eyes, eyebrows, nose, mouth, and chin (13 in totall). You must download shape_predictor_68_face_landmarks.dat and put it in Code folder to make the code work. You can download this file from [here](https://github.com/italojs/facial-landmarks-recognition)

### e) Optical Flow and Landmarks Tracking

### f) Curve Fitting

### g) one vs all classification




## Test the code

```
%cd Code
!python Learning_Algorithm.py
```
