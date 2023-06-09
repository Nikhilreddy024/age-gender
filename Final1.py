#without matching recognised faces

#add fps and inference time and try out all combinations.
#try to check every 5th frame for more speed rather using all frames.

import cv2
from deepface import DeepFace
import time
frame_count = 0
start_time = time.time()
# Load the video
video = cv2.VideoCapture(r'C:\Users\NHI615\Documents\age-gender\mixkit-women-walking-through-fashion-mall-shopping-9060-medium.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 25.0, (640, 480))
backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe'
]

# Load face detection classifier
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    faces = DeepFace.extract_faces(img_path = frame, 
        target_size = (224, 224), 
        detector_backend = backends[4],enforce_detection=False)
    
    if len(faces) == 0:
        continue
    
    for i in range(len(faces)):
        face=faces[i]
        # Extract the first face from the grayscale frame
        x, y, w, h = face['facial_area']['x'],face['facial_area']['y'],face['facial_area']['w'],face['facial_area']['h']
        face = frame[y:y+h, x:x+w]
    
    # Predict age of the face
        result = DeepFace.analyze(face, actions=['age','gender'],enforce_detection=False)
    #age = result['age']
    # Predict age of the face
        age = result[0]['age']
        gender=result[0]['gender']
        

    
    # Draw rectangle and age on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Age: ' + str(int(age)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, 'Gender: ' + str(gender), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    
    # Write the frame to the output video
        out.write(frame)
        height, width = frame.shape[:2]    
        resized_img = cv2.resize(frame, (int(width/3), int(height/3)), interpolation=cv2.INTER_LINEAR)    
    
    
    # Display the frame
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
