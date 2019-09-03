import libjevois as jevois
import numpy as np
import cv2
import cv2 as cv

MARK_RED = 1
MARK_BLUE = 2
MARK_YELLOW = 3
MARK_ORANGE = 4

class ImavMailbox:

    def __init__(self):
        self.alt = 0 # in mm from AP

        # initial color thresholds
        self.red_th = (np.array([141, 76, 0]), np.array([163, 103, 255])) # set to None for splitted red
        self.red1_th = (np.array([163, 19, 0]), np.array([179, 255, 255]))
        self.red2_th = (np.array([0, 50, 0]), np.array([18, 255, 255]))
        self.blue_th = (np.array([100, 138, 0]), np.array([130, 180, 255]))
        self.yellow_th = (np.array([45, 7, 0]), np.array([95, 40, 255]))
        self.orange_th = (np.array([141, 61, 0]), np.array([163, 76, 255]))

        # Define square approximation parameters
        self.width_height_ratio = 0.6
        self.area_occupancy_ratio = 0.6

        self.save = None # save current image

    def processNoUSB(self, inframe):
        img = inframe.getCvBGR()
        self.processImage(img) # no need to process returned data

    def process(self, inframe, outframe):
        img = inframe.getCvBGR()
        detect = self.processImage(img)
        #inframe.done() # release input image
        for mark in detect.values():
            box = cv2.boxPoints(mark)
            ctr = np.array(box).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(img, [ctr], -1, (0, 255, 0), 4)
        outframe.sendCv(img)

    def processImage(self, img):
        '''
        process a single image
        return a dict with detected featured
        '''
        if self.save is not None:
            cv2.imwrite(self.save, img)
            jevois.LINFO(self.save)
            self.save = None

        blur = cv2.GaussianBlur(img,(5,5),0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        detect = {} # dict of detected objects

        # red
        mask_red = None
        if self.red_th is not None:
            mask_red = cv2.inRange(hsv, self.red_th[0], self.red_th[1])
        else:
            mask_red = cv2.inRange(hsv, self.red1_th[0], self.red1_th[1]) + cv2.inRange(hsv, self.red2_th[0], self.red2_th[1])
        ret = self.apply_filters(hsv, mask_red)
        if ret is not None:
            detect[MARK_RED] = ret
            self.send_message(MARK_RED, ret)

        # blue
        mask_blue = cv2.inRange(hsv, self.blue_th[0], self.blue_th[1])
        ret = self.apply_filters(hsv, mask_blue)
        if ret is not None:
            detect[MARK_BLUE] = ret
            self.send_message(MARK_BLUE, ret)

        # yellow
        mask_yellow = cv2.inRange(hsv, self.yellow_th[0], self.yellow_th[1])
        ret = self.apply_filters(hsv, mask_yellow)
        if ret is not None:
            detect[MARK_YELLOW] = ret
            self.send_message(MARK_YELLOW, ret)

        # orange (assuming only one in image)
        mask_orange = cv2.inRange(hsv, self.orange_th[0], self.orange_th[1])
        ret = self.apply_filters(hsv, mask_orange)
        if ret is not None:
            detect[MARK_ORANGE] = ret
            self.send_message(MARK_ORANGE, ret)

        return detect

    def apply_filters(self, hsv, mask):
        '''
        apply filters for detection of a particular color
        '''
        kernel = np.ones((5,5),np.uint8) # create convolution
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # opening
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts) > 0:
            cnt = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            _, (w, h), _ = rect
            return rect
            similarity = min(w, h) / max(w, h)
            area_percentage = cv2.contourArea(cnt) / (w * h)
            if 350 > max(w, h) and min(w, h) > 30 and similarity > self.width_height_ratio and area_percentage > self.area_occupancy_ratio:
                return rect
            else:
                return None

    def send_message(self, mark, pos):
        '''
        send message over serial link to AP
        '''
        (x, y), (w, h), _ = pos
        # TODO apply intrinsec matrix here to send result in image frame
        jevois.sendSerial("N2 {} {:.2f} {:.2f} {:.2f} {:.2f}".format(mark, x, y, w, h))

    def parseSerial(self, cmd):
        str_list = cmd.split(' ')
        if len(str_list) == 2 and str_list[0] == "alt" and str_list[1].isdigit():
            self.alt = int(str_list[1])
            return "OK"
        elif len(str_list) == 2 and str_list[0] == "save":
            self.save = "/jevois/data/images/{}.png".format(str_list[1])
            return self.save
        return "ERR"

    def supportedCommands(self):
        return "alt - set alt in mm"

