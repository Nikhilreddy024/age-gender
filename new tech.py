#extract faces first and then use find()





import cv2
from deepface import DeepFace
import os

# Load the video
video = cv2.VideoCapture(r'C:\Users\NHI615\Documents\age-gender\mixkit-women-walking-through-fashion-mall-shopping-9060-medium.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 25.0, (640, 480))
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

# Folder to save faces in the face database
face_db_folder = r'C:\Users\NHI615\Documents\database'
os.makedirs(face_db_folder, exist_ok=True)

# Dictionary to store face embeddings and associated age and gender
face_data = {}

while video.isOpened():
    # Read frame from video
    ret, frame = video.read()
    
    if not ret:
        break
    
    faces = DeepFace.find(img_path=frame, db_path=face_db_folder, detector_backend=backends[4])
    
    if len(faces) == 0:
        continue

    
    for face in faces:
        print(face)
        if 'box' not in face:
            continue

        x, y, w, h = face['box']
        face_img = frame[y:y+h, x:x+w]
        
        face_embedding = DeepFace.represent(face_img, model_name='Facenet512', enforce_detection=False)
        
        if face_embedding not in face_data:
            # Perform age and gender prediction only for new faces
            result = DeepFace.analyze(face_img, actions=['age', 'gender'], enforce_detection=False)
            age = [result][0]['age']
            gender = result[0]['gender']
            face_data[face_embedding] = {'age': age, 'gender': gender}
            
            # Save the image of the new face in the face database folder
            image_name = f"face_{len(face_data)}.jpg"
            image_path = os.path.join(face_db_folder, image_name)
            cv2.imwrite(image_path, face_img)
            
        else:
            # Retrieve age and gender from stored face data
            age = face_data[face_embedding]['age']
            gender = face_data[face_embedding]['gender']
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Age: ' + str(int(age)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, 'Gender: ' + str(gender), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    out.write(frame)
    height, width = frame.shape[:2]    
    resized_img = cv2.resize(frame, (int(width/3), int(height/3)), interpolation=cv2.INTER_LINEAR)
    
    cv2.imshow('frame', resized_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
video.release()
out.release()
cv2.destroyAllWindows()
