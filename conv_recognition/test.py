import cv2 
from face_detection import RetinaFace

path = 'data/scene1.mp4'
cap = cv2.VideoCapture(path)

detector = RetinaFace(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb)

    for box, landmarks, score in faces:
        if score < 0.9:
            continue
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])

        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
