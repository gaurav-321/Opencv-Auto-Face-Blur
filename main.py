import cv2
import sys
from vidgear.gears import WriteGear

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
output_params = {"-vcodec": "h264", "-crf": 0,
                 "-preset": "fast"}  # define (Codec,CRF,preset) FFmpeg tweak parameters for writer
video_capture = cv2.VideoCapture("test/test_1.mp4")
last = []

writer = WriteGear(output_filename='output/Output.mp4', compression_mode=True, logging=True, **output_params)
while video_capture.read()[0]:
    try:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )
        if len(faces) == 0:
            faces = [last[-1]]
        for cord in faces:
            (x, y, w, h) = cord
            last.append(cord)
            sub_face = frame[y:y + h, x:x + w]
            sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)
            frame[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face

        cv2.imshow('Video', frame)
        writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
writer.close()
