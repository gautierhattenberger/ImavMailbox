import cv2
from DetectMailbox import MailboxDetector
import time
import numpy as np

video = cv2.VideoCapture(1)
time.sleep(2)
#mailbox_red = MailboxDetector([[[163, 173, 10],[179, 255, 245]],[[0, 173, 10],[9, 255, 245]]])
mailbox_red = MailboxDetector([[[0, 173, 0],[9, 255, 255]],[[163, 173, 0],[179, 255, 255]]])
mailbox_blue = MailboxDetector([[[109, 176, 0],[145, 241, 255]]])
mailbox_yellow = MailboxDetector([[[21, 195, 0],[45, 255, 255]]])

video.set(3,640) #width
video.set(4,480) #height
video.set(5,8.3) #fps

while True:
    ret, img = video.read()

    res = mailbox_red.detect(img)
    if res is not None:
        box = cv2.boxPoints(res)
        ctr = np.array(box).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0, 255, 0), 4)

    res = mailbox_blue.detect(img)
    if res is not None:
        box = cv2.boxPoints(res)
        ctr = np.array(box).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0, 255, 0), 4)

    res = mailbox_yellow.detect(img)
    if res is not None:
        box = cv2.boxPoints(res)
        ctr = np.array(box).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0, 255, 0), 4)

    cv2.imshow('frame',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

