#face detection project using openCV
#opencv = open source computer vision library

import cv2

#load pre-trained haar cascade model

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(2, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: cannot open webcam")
    exit()

    print("press 'q' to quit")
    print("press 's' to save image")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grabe frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Face Detected : {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Detection system", frame)

    key=cv2.waitKey(1) & 0xFF

    #prees'q'to exit
    if key == ord('s'):
        cv2.imwrite("captured_face.jpg", frame)
        print("Image saved successfully!")

        #prees 'q' to exit
        if key == ord('q'):
            break

     #Release Resources

    cap.release()
    cv2.destroyAllWindows()
