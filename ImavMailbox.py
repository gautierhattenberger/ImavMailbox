import libjevois as jevois
import numpy as np
import cv2
import cv2 as cv
from DetectMailbox import MailboxDetector

MARK_RED = 1
MARK_BLUE = 2
MARK_YELLOW = 3
MARK_ORANGE = 4

class ImavMailbox:

    def __init__(self):
        self.alt = 0 # in mm from AP

        # initial color thresholds
        self.mailbox_red = MailboxDetector([[[0, 173, 0],[9, 255, 255]],[[163, 173, 0],[179, 255, 255]]])
        self.mailbox_blue = MailboxDetector([[[109, 176, 0],[145, 241, 255]]])
        self.mailbox_yellow = MailboxDetector([[[21, 195, 0],[45, 255, 255]]])
        self.mailbox_orange = MailboxDetector([[[141, 61, 0],[163, 76, 255]]])

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

        detect = {} # dict of detected objects

        # red
        ret = self.mailbox_red.detect(img)
        if ret is not None:
            detect[MARK_RED] = ret
            self.send_message(MARK_RED, ret)

        # blue
        ret = self.mailbox_blue.detect(img)
        if ret is not None:
            detect[MARK_BLUE] = ret
            self.send_message(MARK_BLUE, ret)

        # yellow
        ret = self.mailbox_yellow.detect(img)
        if ret is not None:
            detect[MARK_YELLOW] = ret
            self.send_message(MARK_YELLOW, ret)

        # orange (assuming only one in image)
        ret = self.mailbox_orange.detect(img)
        if ret is not None:
            detect[MARK_ORANGE] = ret
            self.send_message(MARK_ORANGE, ret)

        return detect

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

