This demo is used to identify faces and predict the age and gender of the person.

The program
1) "main" - uses opencv_face_detectors to identify faces and deepface to predict age and gender. 

2)"Final0" - uses opencv_face_detectors to identify faces and age_net.caffemodel,gender_net.caffemodel to predict age and gender.

3)"Final1" - uses deepface to identify faces and to predict age and gender. But without matching
   
   "Matching" - Instead of predict age and gender of faces identified in each frame, we convert the faces identified into an embeddings and store them with age and gender prediction values. so that in every new frame when a face is identified it is checked in the data to avoid repeating predictions.

4)"Final2" - uses deepface to identify faces and to predict age and gender. with matching of recognised faces.

Matching of faces from previous frames increased the fps and decreased inference time.

Out of all available models , 'retinaface' to identify faces provides more accurate results but very slow
and 'FaceNet512' is more accurate for conversion of faces into embeddings.