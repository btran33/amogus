import os
import cv2 as cv2
from amongus import Amongus_Image
from affine_transform import Affine_Transform
from video_result import Amongus_Video
import time
# Referencing this code for face capture: https://github.com/shantnu/Webcam-Face-Detect/blob/master/webcam.py

if not os.path.isdir('output'):
    os.makedirs('output')
video_name = f'output/final_output.mp4'
img = cv2.VideoCapture(video_name)

# Start Capture
cap = cv2.VideoCapture(0)

stop = False
capped = False # Has the face been captured yet?
generated = False # Has the video been generated yet?
cascPath = f'./data/haarcascade_frontalface_default.xml'
data_dir = 'data'
amongus_dir = data_dir + '/amongus'
among_us_faces = [cv2.cvtColor(cv2.imread(f'{amongus_dir}/{file}', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGBA)[140:650, 175:650] # resize config
            for file in os.listdir(amongus_dir)]
faceCascade = cv2.CascadeClassifier(cascPath)
while cap.isOpened():
    # Face detection portion
    # ret, frame = video_capture.read()
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    if not capped:
        time.sleep(1)
        x, y, w, h = faces[0]
        # capture = frame
        # capture = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('data/webcam/capture.jpg', capture)
        bbx = [list(faces[0])]
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # face_bbx = frame[y:y+w, x:x+h]
        capture = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite('data/webcam/bbx.jpg', capture)
        capped = True

    # Display the resulting frame
    # cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Video overlay portion
    # Capture frame-by-frame
    # ret, frame = cap.read()
    
    # Generate the among us video
    if not generated:
        amongus_img = Amongus_Image(capture, among_us_faces, bbx)
        amongus_img.generate_amongus()
        affine_transform = Affine_Transform(amongus_img)
        amongus_video = Amongus_Video(affine_transform, lambda x: (12**x-1)/11)
        amongus_video.generate_video()
        generated = True
        break

img.release()
cap.release()
cv2.destroyAllWindows()
