#!/usr/bin/env python3
import numpy as np
import cv2
import time


def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("`{}` not cannot be loaded".format(image_path))
    return img

def draw_matches(matches, img_left, img_right, verbose=False):
    # Determine the max height
    height = max(img_left.shape[0], img_right.shape[0])
    # Width is the two images side-by-side
    width = img_left.shape[1] + img_right.shape[1]

    img_out = np.zeros((height, width, 3), dtype=np.uint8)
    # Place the images in the empty image 
    img_out[0:img_left.shape[0], 0:img_left.shape[1], :] = img_left
    img_out[0:img_right.shape[0], img_left.shape[1]:, :] = img_right

    # The right image coordinates are offset since the image is no longer at (0,0)
    ow = img_left.shape[1]
   
    #Draw a line between the matched pairs in green
    for p1,p2 in matches:
        p1o = (int(p1[1]), int(p1[0]))
        p2o = (int(p2[1] + ow), int(p2[0]))
        color = list(np.random.random(size=3) * 256)
        cv2.line(img_out, p1o, p2o, color, thickness=2)

    if verbose:
        print("Press enter to continue ... ")
        cv2.imshow("matches", img_out)
        cv2.waitKey(0)

    cv2.imwrite("matches.png", img_out)

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.mins = self.secs // 60
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: {:.2f} s ({:.4f} ms)'.format(self.secs, self.msecs))
