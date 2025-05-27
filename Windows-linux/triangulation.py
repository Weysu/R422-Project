import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

# 1. Lecture des paramètres de calibration depuis les fichiers
def lire_matrice(fichier, mot_cle):
    with open(fichier, 'r') as f:
        lignes = f.readlines()
    idx = [i for i, line in enumerate(lignes) if mot_cle in line][0]
    data = [list(map(float, lignes[idx + i + 1].split())) for i in range(3)]
    return np.array(data)

def lire_distorsion(fichier):
    with open(fichier, 'r') as f:
        lignes = f.readlines()
    idx = [i for i, line in enumerate(lignes) if 'DISTORTION COEFFICIENTS' in line][0]
    coeffs = list(map(float, lignes[idx + 1].split()))
    return np.array(coeffs)

def lire_rotation_translation(fichier):
    with open(fichier, 'r') as f:
        lignes = f.readlines()
    idx_r = [i for i, line in enumerate(lignes) if 'ROTATION MATRIX' in line][0]
    R = np.array([list(map(float, lignes[idx_r + i + 1].split())) for i in range(3)])
    
    idx_t = [i for i, line in enumerate(lignes) if 'TRANSLATION VECTOR' in line][0]
    T = np.array([list(map(float, lignes[idx_t + i + 1].split())) for i in range(3)])
    return R, T.reshape((3,1))

# Fichiers de calibration
fichier_cam1 = 'camera1_calibration.txt'
fichier_cam2 = 'camera2_calibration.txt'
fichier_stereo = 'stereo_calibration.txt'

mtx1 = lire_matrice(fichier_cam1, 'CAMERA MATRIX')
mtx2 = lire_matrice(fichier_cam2, 'CAMERA MATRIX')
dist1 = lire_distorsion(fichier_cam1)
dist2 = lire_distorsion(fichier_cam2)
R, T = lire_rotation_translation(fichier_stereo)

# 2. Matrices de projection
P1 = mtx1 @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = mtx2 @ np.hstack((R, T))

# 3. Fonction de triangulation
def triangulate_points(matches, kp1, kp2, P1, P2):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    points_4d = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T

# 4. Chargement des images et matching
images_left = sorted(glob.glob('Photo/camera1/*.jpg'))
images_right = sorted(glob.glob('Photo/camera2/*.jpg'))

img1 = cv.imread(images_left[0], cv.IMREAD_GRAYSCALE)
img2 = cv.imread(images_right[0], cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)[:50]

# 5. Triangulation
points_3d = triangulate_points(matches, kp1, kp2, P1, P2)

# 6. Inversion Y <-> Z pour repère visuel
points_3d_corrected = points_3d.copy()
points_3d_corrected[:, [1, 2]] = points_3d_corrected[:, [2, 1]]  # échange Y et Z

# 7. Visualisation 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Ajustez les axes selon votre scène
ax.set_xlim3d(-100, 300)
ax.set_ylim3d(1750, 2250)  # Z est maintenant en Y
ax.set_zlim3d(-200, 200)

ax.scatter(points_3d_corrected[:,0], points_3d_corrected[:,1], points_3d_corrected[:,2],
           c='r', marker='o', s=20)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Z (mm)')  # Z affiché en vertical
ax.set_zlabel('Y (mm)')  # Y est profondeur

ax.set_title('Reconstruction 3D corrigée (axes Z/Y échangés)')
plt.tight_layout()
plt.show()

# 8. Sauvegarde au format .txt
np.savetxt('reconstruction_3d_corrected.txt', points_3d_corrected,
           fmt='%.6f', header='X(mm) Z(mm) Y(mm)', comments='')

print("Résultats sauvegardés dans reconstruction_3d_corrected.txt")
