#!/usr/bin/python3
import cv2
from DetectMailbox import MailboxDetector
import numpy as np

def boxPoints(pts):
    if int(cv2.__version__[0]) >= 3:
        return cv2.boxPoints(pts)
    else:
        return cv2.cv.BoxPoints(pts)

DEFAULT_IMAGE_VIEWER = "gwenview"
DEFAULT_IMAGE_OUTPUT = "out_detect.png"
DEFAULT_SCALE_FACTOR = 4
DEFAULT_RESOLUTION = 20 # pixels per meter

mailbox_red = MailboxDetector([[163, 173, 0],[9, 255, 255]], 750, color="RED")
mailbox_blue = MailboxDetector([[103, 129, 0],[129, 190, 255]], 1200, color="BLUE")
#mailbox_yellow = MailboxDetector([[21, 195, 0],[45, 255, 255]], 1500, color="YELLOW")
mailbox_yellow = MailboxDetector([[0, 0, 0],[179, 6, 255]], 1500, aspect_ratio_th=0.6, color="YELLOW") # for test image only
mailbox_orange = MailboxDetector([[141, 61, 0],[163, 76, 255]], 500, color="ORANGE")

def find_mailboxes(img, output=None, scale=DEFAULT_SCALE_FACTOR, res=DEFAULT_RESOLUTION):
    out = img.copy()

    scale_factor = pow(res / 1000., 2)
    res = mailbox_red.detect(img, scale_factor)
    if res is not None:
        center = (int(res[0][0]), int(res[0][1]))
        cv2.circle(out, center, 50, (0, 255, 0), 5)
        cv2.putText(out, "RED", (center[0]+60, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)

    res = mailbox_blue.detect(img, scale_factor)
    if res is not None:
        center = (int(res[0][0]), int(res[0][1]))
        cv2.circle(out, center, 50, (0, 255, 0), 5)
        cv2.putText(out, "BLUE", (center[0]+60, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)

    res = mailbox_yellow.detect(img, scale_factor)
    if res is not None:
        center = (int(res[0][0]), int(res[0][1]))
        cv2.circle(out, center, 50, (0, 255, 0), 5)
        cv2.putText(out, "YELLOW", (center[0]+60, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)

    if output is None:
        w, h, _ = img.shape
        img_out = cv2.resize(out, (int(h/scale),int(w/scale)))
        cv2.imshow('frame',img_out)
        while True:
            if cv2.waitKey(-1)  & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(output, out)

if __name__ == '__main__':
    '''
    When used as a standalone script
    '''
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description="Search mailboxes in image")
    parser.add_argument('img', help="image path")
    parser.add_argument("-v", "--viewer", help="program used to open the image", default=DEFAULT_IMAGE_VIEWER)
    parser.add_argument("-nv", "--no_view", help="Do not open image after processing", action='store_true')
    parser.add_argument("-o", "--output", help="output file name", default=None)
    parser.add_argument("-s", "--scale", help="resize scale factor", type=int, default=DEFAULT_SCALE_FACTOR)
    parser.add_argument("-r", "--resolution", help="resolution in pixels per meter", type=float, default=DEFAULT_RESOLUTION)
    args = parser.parse_args()

    img = cv2.imread(args.img)
    find_mailboxes(img, args.output, args.scale, args.resolution)

    if not args.no_view and args.output is not None:
        subprocess.call([args.viewer, args.output])

