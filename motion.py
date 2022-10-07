import cv2
import numpy as np


cam = cv2.VideoCapture(0)
cam.set(3, 720)
cam.set(4, 1080)
cam.set(10, 100)


ret, frame1 = cam.read()
ret, frame2 = cam.read()

while cam.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilate = cv2.dilate(thresh, None, iterations = 3)
    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for element in contours:
        (x, y, w, h) = cv2.boundingRect(element)

        if cv2.contourArea(element) < 700:
            continue
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame1, "Moving", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)


    cv2.drawContours(frame1, contours, -1, (0,0,255), 2)

    cv2.imshow("camera", frame1)
    frame1 = frame2
    ret, frame2 = cam.read()



    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break


