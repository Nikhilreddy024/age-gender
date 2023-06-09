import cv2
from deepface import DeepFace
import json
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
    
    if not ret:
        break
    
    faces = DeepFace.extract_faces(img_path=frame, target_size=(224, 224), detector_backend=backends[4])
    
    if len(faces) == 0:
        continue
    
    for face in faces:
        #x, y, w, h = face['box']
        x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']

        face_img = frame[y:y+h, x:x+w]
        
        face_embedding = tuple(DeepFace.represent(face_img, model_name='Facenet512', enforce_detection=False))
        face_embedding_str = json.dumps(face_embedding)
        if face_embedding_str not in face_data:
            # Perform age and gender prediction only for new faces
            result = DeepFace.analyze(face_img, actions=['age', 'gender'], enforce_detection=False)
            age = result[0]['age']
            gender = result[0]['gender']
            face_data[face_embedding_str] = {'age': age, 'gender': gender}
        else:
            # Retrieve age and gender from stored face data
            age = face_data[face_embedding_str]['age']
            gender = face_data[face_embedding_str]['gender']
        
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
