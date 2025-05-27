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
    print("ERREUR: Impossible de charger les images. Vérifiez les chemins:")
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    exit()

# Initialisation ORB
orb = cv2.ORB_create()

# Détection des points clés
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Appariement
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Affichage des 10 meilleurs matches
result_img = cv2.drawMatches(
    img1, kp1, img2, kp2, 
    matches[:10], 
    None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Conversion BGR vers RGB pour matplotlib
result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

# Affichage
plt.figure(figsize=(15, 8))
plt.imshow(result_img_rgb)
plt.axis('off')
plt.title("Top 10 correspondances ORB entre les images")
plt.show()