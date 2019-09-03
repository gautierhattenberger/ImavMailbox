import cv2
import cv2 as cv
import numpy as np

class MailboxDetector:

    def __init__(self, hsv_th, aspect_ratio_th=0.8, area_th=0.7, size_th=(20,200)):
        self.hsv_th = np.array(hsv_th)
        self.aspect_ratio_th = aspect_ratio_th
        self.area_th = area_th
        self.size_th = size_th
        self.kernel = np.ones((8,8),np.uint8) # create convolution

    def detect(self, img):

        #blur = cv2.GaussianBlur(img,(5,5),0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = None
        for th in self.hsv_th:
            hsv_min, hsv_max = th
            if mask is None:
                mask = cv2.inRange(hsv, hsv_min, hsv_max)
            else:
                mask += cv2.inRange(hsv, hsv_min, hsv_max)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # opening
        #cv2.imshow('mask',mask)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        #self.draw_all(img.copy(),cnts)
        best_res = None
        best_score = 0.
        for cnt in cnts:
            rect = cv2.minAreaRect(cnt)
            _, (w, h), _ = rect
            min_wh = min(w, h)
            max_wh = max(w, h)
            if min_wh < self.size_th[0] and max_wh > self.size_th[1]:
                #print("not correct size")
                continue # too small or too big
            similarity = min_wh / max_wh
            if similarity < self.aspect_ratio_th:
                #print("not square")
                continue # not enough square
            area_ratio = cv2.contourArea(cnt) / (w * h) 
            if area_ratio < self.area_th:
                #print("not good ratio")
                continue # not enough full of color
            score = area_ratio * similarity * (w*h)
            #print(score,best_score)
            if score > best_score:
                best_score = score
                best_res = rect
        return best_res

    def draw_all(self, img, cnts):
        for cnt in cnts:
            box = cv2.boxPoints(cv2.minAreaRect(cnt))
            ctr = np.array(box).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(img, [ctr], -1, (0, 255, 0), 4)
        #cv2.imshow('contour',img)
