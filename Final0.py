#opencv models for face,age and gender

import cv2


def getFaceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    faceBoxes = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.5:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, faceBoxes


faceProto = r"C:\Users\NHI615\Documents\age-gender\opencv_face_detector.pbtxt"
faceModel = r"C:\Users\NHI615\Documents\age-gender\opencv_face_detector_uint8.pb"

ageProto = r"C:\Users\NHI615\Documents\age-gender\age_deploy.prototxt"
ageModel = r"C:\Users\NHI615\Documents\age-gender\age_net.caffemodel"

genderProto = r"C:\Users\NHI615\Documents\age-gender\gender_deploy.prototxt"
genderModel = r"C:\Users\NHI615\Documents\age-gender\gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

video = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output6.mp4', fourcc, 25.0, (int(video.get(3)), int(video.get(4))))




padding = 20

while True:
    hasFrame, vidFrame = video.read()

    if not hasFrame:
        cv2.waitKey()
        break

    frame, faceBoxes = getFaceBox(faceNet, vidFrame)

    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
               max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        labelGender = "{}".format("Gender : " + gender)
        labelAge = "{}".format("Age : " + age + "Years")
        cv2.putText(frame, labelGender, (faceBox[0], faceBox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, labelAge, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
    height, width = frame.shape[:2]    
    resized_img = cv2.resize(frame, (int(width/3), int(height/3)), interpolation=cv2.INTER_LINEAR)    
    cv2.imshow("Age-Gender Detector", resized_img)
    out.write(frame) 
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()
