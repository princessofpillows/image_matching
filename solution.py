#!/usr/bin/env python3
import cv2, math, pywt
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
from skimage.transform import warp, AffineTransform
from operator import itemgetter
from harris import harris, draw_corners
from utils import read_image, draw_matches, Timer


def ransac(pts1, pts2, img_l, img_r, max_iters=1000, epsilon=1):
    best_matches = []
    # Number of samples
    N = 4
    
    for i in range(max_iters):
        # Get 4 random samples from features
        idx = np.random.randint(0, len(pts1) - 1, N)
        src = pts1[idx]
        dst = pts2[idx]

        # Calculate the homography matrix H
        H = cv2.getPerspectiveTransform(src, dst)
        Hp = cv2.perspectiveTransform(pts1[None], H)[0]

        # Find the inliers by computing the SSD(p',Hp) and saving inliers (feature pairs) that are SSD(p',Hp) < epsilon
        inliers = []
        for i in range(len(pts1)):
            ssd = np.sum(np.square(pts2[i] - Hp[i]))
            if ssd < epsilon:
                inliers.append([pts1[i], pts2[i]])
        
        # Keep the largest set of inliers and the corresponding homography matrix
        if len(inliers) > len(best_matches):
            best_matches = inliers
    
    return best_matches

# Plotting tool for convenience
def plot(plots):
    fig = plt.figure()
    N = len(plots)
    x = math.ceil(math.sqrt(N))
    y = math.ceil(math.sqrt(N))
    for i in range(N):
        fig.add_subplot(x,y,i+1).imshow(plots[i])
    plt.show()

def mops(img, truth, win_size, h, w, r):
    H, W = img.shape
    offset = win_size // 2
    
    # Draw line for angle of gradient (for debugging)
    #length = 150
    #h2 =  int(h - length * math.cos(math.radians(r)))
    #w2 =  int(w - length * math.sin(math.radians(r)))
    #cv2.line(img, (w, h), (w2, h2), (0,255,0), 2)

    # Rotate image s.t. gradient angle of feature is origin
    M = cv2.getRotationMatrix2D((w, h), -1*r, 1)
    img_rot = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_NEAREST)

    # Get 40x40 window around feature
    win = img_rot[h-offset:h+offset, w-offset:w+offset]

    # Size (s,s) of each sample region
    s = win_size // 8
    # Prefiltering (N,N) -> (N//s, N//s)
    i = 0
    rows = []
    while i < win_size:
        j = 0
        cols = []
        while j < win_size:
            # Sample (s,s) region from window
            sample = win[i:i+s, j:j+s]
            # Downsample (s,s) region to a single value
            sample = np.sum(sample) / (s*s)
            cols.append(sample)
            j += s
        rows.append(cols)
        i += s
    
    feature = np.array(rows)
    # Normalize
    feature = (feature - np.mean(feature)) / np.std(feature)
    # Haar wave transform
    coeffs = pywt.dwt2(feature, 'haar')
    feature = pywt.idwt2(coeffs, 'haar')
    #plot([img, img_rot, truth, win, feature])
    return feature

def get_matches(ft1, ft2, left_match, right_match, img_left, img_right, max_matches, win_size):
    potential_matches = []
    offset = win_size // 2

    # Feature descriptor with brute-force matching
    for h1, w1, r1 in ft1:
        
        # Copy image as to not modify it by reference
        img_left_tmp = np.copy(img_left)

        # Copy coordinates incase changed by border
        h1_tmp = h1
        w1_tmp = w1

        # Get 40x40 window around feature
        win_left = img_left[h1-offset:h1+offset, w1-offset:w1+offset]

        # Handle border points
        if win_left.shape != (win_size, win_size):
            diff_h, diff_w = np.subtract(win_left.shape, (win_size, win_size))
            p_h = abs(diff_h // 2)
            p_w = abs(diff_w // 2)
            img_left_tmp = cv2.copyMakeBorder(img_left_tmp, p_h, p_h, p_w, p_w, cv2.BORDER_REFLECT)
            h1 += p_h
            w1 += p_w
            win_left = img_left_tmp[h1-offset:h1+offset, w1-offset:w1+offset]
        
        # Run multiscale oriented patches descriptor
        feature_left = mops(img_left_tmp, win_left, win_size, h1, w1, r1)

        lowest_dist = math.inf
        potential_match = ()
        for h2, w2, r2 in ft2:

            # Copy image as to not modify it by reference
            img_right_tmp = np.copy(img_right)

            # Copy coordinates incase changed by border
            h2_tmp = h2
            w2_tmp = w2

            # Get 40x40 window around feature
            win_right = img_right[h2-offset:h2+offset, w2-offset:w2+offset]

            # Handle border points
            if win_right.shape != (win_size, win_size):
                diff_h, diff_w = np.subtract(win_right.shape, (win_size, win_size))
                p_h = abs(diff_h // 2)
                p_w = abs(diff_w // 2)
                img_right_tmp = cv2.copyMakeBorder(img_right_tmp, p_h, p_h, p_w, p_w, cv2.BORDER_REFLECT)
                h2 += p_h
                w2 += p_w
                win_right = img_right_tmp[h2-offset:h2+offset, w2-offset:w2+offset]

            # Run multiscale oriented patches descriptor
            feature_right = mops(img_right_tmp, win_right, win_size, h2, w2, r2)
            
            # Check distance between features
            curr_dist = np.linalg.norm(feature_left - feature_right)
            if curr_dist < lowest_dist:
                lowest_dist = curr_dist
                potential_match = ([h1_tmp, w1_tmp, r1], [h2_tmp, w2_tmp, r2], curr_dist)
        
        potential_matches.append(potential_match)
        
    # Sort matches from smallest distance up
    matches = sorted(potential_matches, key=itemgetter(2))
    for match in matches:
        # Ensure no duplicates
        if match[0][0:2] not in left_match and match[1][0:2] not in right_match:
            # Add to matches
            left_match.append(match[0][0:2])
            right_match.append(match[1][0:2])
            # Remove from potential matches
            ft1.remove(tuple(match[0]))
            ft2.remove(tuple(match[1]))
    
    # Recursively keep going until every point has a match
    while(len(left_match) < max_matches and len(right_match) < max_matches):
        print('anotha one')
        get_matches(ft1, ft2, left_match, right_match, img_left, img_right, max_matches, win_size)

    return np.array(left_match, dtype=np.float32), np.array(right_match, dtype=np.float32)

def main(args):
    # Load the image
    img_left_clr   = read_image(args.image)
   
    # Perform a some transforms on it for feature matching 
    tform = AffineTransform(scale=(0.8, 0.8), rotation=0.2, translation=(20, -10))
    img_right_clr = warp(img_left_clr, tform.inverse, output_shape=img_left_clr.shape[:2])
    img_right_clr = np.uint8(img_right_clr * 255) 

    # Convert to grayscale
    img_left = cv2.cvtColor(img_left_clr, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right_clr, cv2.COLOR_RGB2GRAY)

    print("Getting the features from the Harris Detector")
    ftL = harris(img_left, sigma=3, threshold=0.01)
    draw_corners(ftL, img_left_clr, 'corners_left')
    ftR = harris(img_right, sigma=3, threshold=0.01)
    draw_corners(ftR, img_right_clr, 'corners_right')
    
    print("  -- Number of features (left): ", len(ftL))
    print("  -- Number of features (right): ", len(ftR))
    print("Finding the best matches between images")
    with Timer(verbose=True) as t:
        max_matches = min(len(ftL), len(ftR))
        ptsL,ptsR = get_matches(ftL, ftR, [], [], img_left, img_right, max_matches, win_size=args.win_size)
        
        print(" -- Number of matches = ", len(ptsL))
        assert len(ptsL) == len(ptsR)

    print("Performing RANSAC")
    matches = ransac(ptsL,ptsR, img_left, img_right, args.max_iters, args.epsilon)
    print(" -- Number of pruned matches = ", len(matches))
    
    draw_matches(matches, img_left_clr, img_right_clr)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("image", 
                        default="imgs/left.png",
                        help="Path to left image") 
    
    parser.add_argument("-w", 
                        "--win_size", 
                        default=50,
                        type=int,
                        help="Window size for your feature detector algorithm") 
    
    parser.add_argument("-i", 
                        "--max_iters", 
                        default=1000,
                        type=int,
                        help="Maximum iterations to perform RANSAC") 
    
    parser.add_argument("-e", 
                        "--epsilon", 
                        default=100,
                        type=float,
                        help="SSD epsilon threshold for RANSAC") 
   
    parser.add_argument("-v", 
                        "--verbose", 
                        help="Turn on debugging statements",
                        action="store_true") 

    args = parser.parse_args()
    main(args)
