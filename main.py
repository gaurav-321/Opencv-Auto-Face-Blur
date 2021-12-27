import traceback

import cv2

input_file = r"test/test_1.mp4"
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
SCALE = 0.2  # lowering it increase the performance but decrease accuracy
RATIO = round(1/SCALE) # lowering it increase the performance but decrease accuracy
cap = cv2.VideoCapture(input_file)
ret, frame = cap.read()  # Get one ret and frame
h, w, _ = frame.shape  # Use frame to get width and height
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("output/Output.mp4", fourcc, 24, (w, h))
faces_loc = []
while ret:
    try:
        ret, frame = cap.read()
        frame_small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        faces = cascade.detectMultiScale(
            cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(60, 60)
        )

        if len(faces) == 0 and len(faces_loc) > 2:
            faces = faces_loc[-2:]
        for (x, y, w, h) in faces:
            faces_loc.append((x, y, w, h))
            sub_face = frame[y * RATIO:y * RATIO + h * RATIO, x * RATIO:x * RATIO + w * RATIO]
            sub_face = cv2.blur(sub_face, (50, 50))
            frame[y * RATIO:y * RATIO + h * RATIO, x * RATIO:x * RATIO + w * RATIO] = sub_face
        frame_small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        cv2.imshow('Video', frame_small)
        writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()
writer.release()
