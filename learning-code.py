# How to do Face Detector with Pyhton
# - Step 1: Get a Crap-Load of faces
# - Step 2: Make them all black and white
# - Step 3: Train the algorithm to detect faces

import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# trained_face_data = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# To capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    # rectangle_colors = (randrange(0, 255), randrange(0, 255), randrange(0, 255))

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detector with Python', frame)
    key = cv2.waitKey(1)
    # one millisecond

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
webcam.release()
print("")
    


# VideoCapture('file.mp4')
# VideoCapture(0) -> use webcam

'''
# Choose an image to detect faces in...
# img = cv2.imread('images/holland-robert.jpg')
# img = cv2.imread('images/robert.jpeg')
# img = cv2.imread('images/people.jpeg')

# Must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# [[ 78  15 134 134]]
print(face_coordinates)

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(0, 255), randrange(0, 255), randrange(0, 255)), 2)
 
# Display the image with the faces
cv2.imshow('Face Detector with Python', img)
cv2.waitKey()


print("Code Completed...")
'''