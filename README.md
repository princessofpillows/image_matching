
# Feature Matching

This project uses Harris Corners (non-scale invariant) with non-maximum suppression to detect features, the feature descriptor from MOPS (multiscale oriented patches descriptor) to select matches between features, and RANSAC (random sample consensus) to prune matches.

The Harris Corners takes an input image, gets the derivatives of that image, dx and dy, then creates the H matrix from these values. It then creates the Harris Operator R by taking the determinate of H / trace of H. Non-maximum suppresion occurs on R, and the output is returned as [H, W, theta], where theta is the angle of the gradient for each point (H, W) where R > min_threshold.

For each feature point p, MOPS uses pH, pW, ptheta and rotates the entire image by ptheta before taking a (N,N) window around p (border points are accounted for). The rotation is to ensure the gradient angle for each feature is pointing in the same direction, which allows for simplified matching. Sampling occurs by taking mini-windows of size N/s within window p, where s is a hyperparameter for size vs number of mini-windows. Then for each mini-window, it is downsampled to a single point, and placed in a new image of size (N/s)x(N/s). This image is normalized and fed through a Haar wavelet, before being used for feature matching.

The distance function for the feature descriptor is Euclidean. Matching is done by taking the best matches for each point, sorting by distance, and then removing each match from potential matches to ensure there are no duplicates. If there are remaining potential matches, the feature selection is called recursively until each feature point is assigned a match.

RANSAC is performed on every match to remove outliars and keep valid matches.

Example output shown below.

![alt text](https://raw.githubusercontent.com/jpatts/feature_matching/master/matches_test.png)
![alt text](https://raw.githubusercontent.com/jpatts/feature_matching/master/matches_island.png)
