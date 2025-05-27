import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

images_left = sorted(glob.glob('Photo/camera1/camera_1.jpg'))
images_right = sorted(glob.glob('Photo/camera2/camera_2.jpg'))

# Initiate ORB detector
orb = cv.ORB_create()

# Charger les images (prend la première image de chaque dossier pour l'exemple)
img1 = cv.imread(images_left[0], cv.IMREAD_GRAYSCALE)
img2 = cv.imread(images_right[0], cv.IMREAD_GRAYSCALE)

# Vérifie que les images sont bien chargées
if img1 is None or img2 is None:
    raise ValueError("Erreur de chargement d'image.")

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3), plt.show()
