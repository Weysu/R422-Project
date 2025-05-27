import cv2
import matplotlib.pyplot as plt

# Chemins des images
img1_path = "Photo/camera1/camera_1.jpg"
img2_path = "Photo/camera2/camera_2.jpg"

# Chargement des images en niveaux de gris
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Vérification du chargement
if img1 is None or img2 is None:
    print("ERREUR: Impossible de charger les images.")
    exit()

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

if des1 is None or des2 is None:
    print("ERREUR: Aucun descripteur trouvé.")
    exit()

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Filtrage avec ratio test
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

# Dessiner les correspondances
result_img = cv2.drawMatches(img1, kp1, img2, kp2, good[:10], None)

# Affichage
result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15, 8))
plt.imshow(result_img_rgb)
plt.axis('off')
plt.title("Top 10 correspondances SIFT entre les images")
plt.show()
