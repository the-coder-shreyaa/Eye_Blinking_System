

import cv2
import time
face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
face_model = cv2.CascadeClassifier(face_path)
eye_model = cv2.CascadeClassifier(eye_path)
cam = cv2.VideoCapture(0)
blink_count = 0
eye_closed_frames = 0
EYE_CLOSED_THRESHOLD = 3  # frames

while True:
    status, frame = cam.read()
    if not status:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_model.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_model.detectMultiScale(roi_gray, 1.2, 5)

        if len(eyes) == 0:
            eye_closed_frames += 1
        else:
            if eye_closed_frames >= EYE_CLOSED_THRESHOLD:
                blink_count += 1
                print("Blink detected")
            eye_closed_frames = 0

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.putText(frame, f"Blinks: {blink_count}",
                 (20, 40),
                 cv2.FONT_HERSHEY_SIMPLEX,
                 1,
                 (0, 255, 255),
                 2)

    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
cv2.destroyAllWindows()