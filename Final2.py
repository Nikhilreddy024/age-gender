#with matching recognised faces


import os
import cv2
from deepface import DeepFace
import json
from scipy.spatial.distance import euclidean
import numpy as np
import time
frame_count = 0
start_time = time.time()
# Load the video
video = cv2.VideoCapture(r'C:\Users\NHI615\Documents\age-gender\mixkit-women-walking-through-fashion-mall-shopping-9060-medium.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 25.0, (640, 480))
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

# Dictionary to store face embeddings and associated age and gender
face_data = {}

while video.isOpened():
    # Read frame from video
    ret, frame = video.read()
    frame_count += 1
    start_inference = time.time()
    if time.time() - start_time >= 1.0:
        fps = frame_count / (time.time() - start_time)
        print(f"FPS: {fps}")
        frame_count = 0
        start_time = time.time()
    
    if not ret:
        break

    
    faces = DeepFace.extract_faces(img_path=frame, target_size=(224, 224), detector_backend=backends[4],enforce_detection=False)
    
    if len(faces) == 0:
        continue
    
    for face in faces:

        print('new face')
        #x, y, w, h = face['box']
        x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']

        face_img = frame[y:y+h, x:x+w]
        
        face_embedding = tuple(DeepFace.represent(face_img, model_name='Facenet512', enforce_detection=False))
        face_embedding_str = json.dumps(face_embedding)
        matched_face = None
        for stored_face_embedding_str in face_data:
            stored_face_embedding = json.loads(stored_face_embedding_str)
            distance = euclidean(np.array(face_embedding[0]['embedding']), np.array(stored_face_embedding[0]['embedding']))
            if distance < 11:  # Adjust the threshold as per your requirements
    
                matched_face = stored_face_embedding_str
                break
        
        if matched_face is None:
            # Perform age and gender prediction only for new faces
            result = DeepFace.analyze(face_img, actions=['age', 'gender'], enforce_detection=False)
            age = result[0]['age']
            gender = result[0]['gender']
            
            # Store the new face embedding and its attributes
            face_data[face_embedding_str] = {'age': age, 'gender': gender}
        else:
            print('matched')
            # Retrieve age and gender from stored face data
            age = face_data[matched_face]['age']
            gender = face_data[matched_face]['gender']


        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Age: ' + str(int(age)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, 'Gender: ' + str(gender), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    out.write(frame)
    height, width = frame.shape[:2]    
    resized_img = cv2.resize(frame, (int(width/3), int(height/3)), interpolation=cv2.INTER_LINEAR)
    
    cv2.imshow('frame', resized_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    end_inference = time.time()
    inference_time = end_inference - start_inference
    print(f"Inference Time: {inference_time} seconds")

# Release everything if job is finished
video.release()
out.release()
cv2.destroyAllWindows()
