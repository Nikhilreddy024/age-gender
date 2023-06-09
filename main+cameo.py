#opencv model for face with deepface for age and gender


import cv2
from deepface import DeepFace
import numpy as np

# Load the video
video = cv2.VideoCapture(r'C:\Users\NHI615\Documents\age-gender\_import_60cc295572f0a0.70228509.mov')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 25.0, (640, 480))

# Load face detection model
modelFile = r"C:\Users\NHI615\Documents\age-gender\opencv_face_detector_uint8.pb"
configFile = r"C:\Users\NHI615\Documents\age-gender\opencv_face_detector.pbtxt"

net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

while video.isOpened():
    # Read frame from video
    ret, frame = video.read()
    
    if not ret:
        break
    
    # Extract blob from the frame and detect faces using the face detection model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)

    #blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    
    # If no face is detected, skip this frame
    if detections.shape[2] == 0:
        continue
    else:
        # Extract the first face from the frame
        box = detections[0, 0, 0, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
        (x, y, w, h) = box.astype("int")
        face = frame[y:y+h, x:x+w]
    
    # Predict age and gender of the face
    
    result = DeepFace.analyze(face, actions=['age','gender'],enforce_detection=False)
    age = result[0]['age']
    gender=result[0]['gender']

    # Draw rectangle and age on the frame
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, 'Age: ' + str(int(age)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, 'Gender: ' + str(gender), (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Write the frame to the output video
    out.write(frame)
    height, width = frame.shape[:2]    
    resized_img = cv2.resize(frame, (int(width/3), int(height/3)), interpolation=cv2.INTER_LINEAR)    
    
    # Display the frame
    cv2.imshow('frame', resized_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
video.release()
out.release()
cv2.destroyAllWindows()
