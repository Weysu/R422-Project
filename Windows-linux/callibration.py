import numpy as np
import cv2 as cv
import glob
import os

def mono_calibration(img_folder, chessboard_size=(9,6), square_size=25, show_corners=False):
    """
    Calibration mono-caméra
    
    Args:
        img_folder (str): Chemin vers les images de calibration (ex: "photos/camera1_te/*.jpg")
        chessboard_size (tuple): Dimensions du damier (coins internes)
        square_size (float): Taille d'un carré en mm
        show_corners (bool): Afficher la détection des coins
    
    Returns:
        mtx (ndarray): Matrice intrinsèque (Camera Matrix)
        dist (ndarray): Coefficients de distortion (Distortion Coefficients)
        img_size (tuple): Taille des images (w, h)
    """
    # Critères d'arrêt pour l'algorithme de sous-pixel
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Préparation des points 3D du damier
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    
    # Stockage des points
    objpoints = []  # Points 3D réels
    imgpoints = []  # Points 2D dans l'image
    img_size = None
    
    # Récupération des images
    images = glob.glob(img_folder)
    print(f"Found {len(images)} images for calibration")
    
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}")
            continue
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  # (width, height)
        
        # Détection des coins du damier
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            objpoints.append(objp)
            
            # Affinage de la position des coins
            corners_refined = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners_refined)
            
            # Visualisation (optionnelle)
            if show_corners:
                cv.drawChessboardCorners(img, chessboard_size, corners_refined, ret)
                cv.imshow('Chessboard Corners', img)
                cv.waitKey(500)
    
    if show_corners:
        cv.destroyAllWindows()
    
    # Calibration de la caméra
    if len(objpoints) < 5:
        raise ValueError(f"Only {len(objpoints)} valid images found. Need at least 5 for calibration.")
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)
    
    # Calcul de l'erreur de reprojection
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"Calibration error: {mean_error/len(objpoints):.3f} pixels")
    
    return mtx, dist, img_size

def stereo_calibration(img_folder1, img_folder2, mtx1, dist1, mtx2, dist2, 
                      chessboard_size=(9,6), square_size=25, show_corners=False):
    """
    Calibration stéréo entre deux caméras
    
    Args:
        img_folder1 (str): Chemin vers les images de la caméra 1
        img_folder2 (str): Chemin vers les images de la caméra 2
        mtx1, dist1: Paramètres de calibration de la caméra 1
        mtx2, dist2: Paramètres de calibration de la caméra 2
        chessboard_size (tuple): Dimensions du damier
        square_size (float): Taille d'un carré en mm
        show_corners (bool): Afficher la détection des coins
    
    Returns:
        R (ndarray): Matrice de rotation entre les caméras
        T (ndarray): Vecteur de translation entre les caméras
        E (ndarray): Matrice essentielle
        F (ndarray): Matrice fondamentale
        img_size (tuple): Taille des images
    """
    # Critères d'arrêt
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Préparation des points 3D du damier
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    
    # Stockage des points pour les deux caméras
    objpoints = []
    imgpoints1 = []  # Points dans la caméra 1
    imgpoints2 = []  # Points dans la caméra 2
    
    # Récupération des images des deux caméras
    images1 = sorted(glob.glob(img_folder1))
    images2 = sorted(glob.glob(img_folder2))
    
    print(f"Found {len(images1)} images for camera 1 and {len(images2)} images for camera 2")
    
    if len(images1) != len(images2):
        raise ValueError("Number of images must be the same for both cameras")
    
    img_size = None
    valid_pairs = 0
    
    for fname1, fname2 in zip(images1, images2):
        img1 = cv.imread(fname1)
        img2 = cv.imread(fname2)
        
        if img1 is None or img2 is None:
            print(f"Warning: Could not read images {fname1} or {fname2}")
            continue
        
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        img_size = gray1.shape[::-1]
        
        # Détection des coins dans les deux images
        ret1, corners1 = cv.findChessboardCorners(gray1, chessboard_size, None)
        ret2, corners2 = cv.findChessboardCorners(gray2, chessboard_size, None)
        
        # On ne garde que les paires où les coins sont détectés dans les deux images
        if ret1 and ret2:
            objpoints.append(objp)
            
            # Affinage des coins
            corners1_refined = cv.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
            corners2_refined = cv.cornerSubPix(gray2, corners2, (11,11), (-1,-1), criteria)
            
            imgpoints1.append(corners1_refined)
            imgpoints2.append(corners2_refined)
            
            valid_pairs += 1
            
            # Visualisation (optionnelle)
            if show_corners:
                cv.drawChessboardCorners(img1, chessboard_size, corners1_refined, ret1)
                cv.drawChessboardCorners(img2, chessboard_size, corners2_refined, ret2)
                
                # Affichage côte à côte
                combined = np.hstack((img1, img2))
                cv.imshow('Stereo Chessboard Detection', combined)
                cv.waitKey(500)
    
    if show_corners:
        cv.destroyAllWindows()
    
    print(f"Found {valid_pairs} valid stereo pairs for calibration")
    
    if valid_pairs < 5:
        raise ValueError(f"Only {valid_pairs} valid stereo pairs found. Need at least 5 for stereo calibration.")
    
    # Calibration stéréo
    flags = cv.CALIB_FIX_INTRINSIC  # Utilise les paramètres intrinsèques déjà calculés
    
    ret, mtx1_new, dist1_new, mtx2_new, dist2_new, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        mtx1, dist1, mtx2, dist2,
        img_size, flags=flags)
    
    # Calcul de l'erreur de calibration stéréo
    print(f"Stereo calibration RMS error: {ret:.3f}")
    
    # Affichage des résultats
    print(f"\nRotation matrix R:")
    print(R)
    print(f"\nTranslation vector T (mm):")
    print(T.flatten())
    print(f"\nBaseline (distance between cameras): {np.linalg.norm(T):.2f} mm")
    
    return R, T, E, F, img_size

def save_calibration(filename, mtx, dist, img_size):
    """Sauvegarde des paramètres de calibration mono dans un fichier txt"""
    with open(filename, 'w') as f:
        f.write("=== MONO CAMERA CALIBRATION PARAMETERS ===\n\n")
        
        f.write("CAMERA MATRIX (Intrinsic Matrix):\n")
        for row in mtx:
            f.write(" ".join([f"{val:12.6f}" for val in row]) + "\n")
        
        f.write("\nDISTORTION COEFFICIENTS:\n")
        f.write(" ".join([f"{val:12.6f}" for val in dist.flatten()]) + "\n")
        
        f.write(f"\nIMAGE SIZE (width x height):\n")
        f.write(f"{img_size[0]} {img_size[1]}\n")
        
        # Paramètres individuels pour faciliter la lecture
        f.write(f"\n=== INDIVIDUAL PARAMETERS ===\n")
        f.write(f"Focal length X (fx): {mtx[0,0]:.6f}\n")
        f.write(f"Focal length Y (fy): {mtx[1,1]:.6f}\n")
        f.write(f"Principal point X (cx): {mtx[0,2]:.6f}\n")
        f.write(f"Principal point Y (cy): {mtx[1,2]:.6f}\n")
        
        if len(dist.flatten()) >= 5:
            f.write(f"Radial distortion k1: {dist[0,0]:.6f}\n")
            f.write(f"Radial distortion k2: {dist[0,1]:.6f}\n")
            f.write(f"Tangential distortion p1: {dist[0,2]:.6f}\n")
            f.write(f"Tangential distortion p2: {dist[0,3]:.6f}\n")
            f.write(f"Radial distortion k3: {dist[0,4]:.6f}\n")
    
    print(f"Mono calibration saved to {filename}")

def save_stereo_calibration(filename, R, T, E, F, img_size):
    """Sauvegarde des paramètres de calibration stéréo dans un fichier txt"""
    with open(filename, 'w') as f:
        f.write("=== STEREO CALIBRATION PARAMETERS ===\n\n")
        
        f.write("ROTATION MATRIX (R):\n")
        for row in R:
            f.write(" ".join([f"{val:12.6f}" for val in row]) + "\n")
        
        f.write("\nTRANSLATION VECTOR (T) [mm]:\n")
        for val in T.flatten():
            f.write(f"{val:12.6f}\n")
        
        f.write("\nESSENTIAL MATRIX (E):\n")
        for row in E:
            f.write(" ".join([f"{val:12.6f}" for val in row]) + "\n")
        
        f.write("\nFUNDAMENTAL MATRIX (F):\n")
        for row in F:
            f.write(" ".join([f"{val:12.6f}" for val in row]) + "\n")
        
        f.write(f"\nIMAGE SIZE (width x height):\n")
        f.write(f"{img_size[0]} {img_size[1]}\n")
        
        # Informations supplémentaires
        f.write(f"\n=== ADDITIONAL INFO ===\n")
        f.write(f"Baseline (distance between cameras): {np.linalg.norm(T):.2f} mm\n")
        
        # Angles de rotation (en degrés)
        import math
        angles = np.array([
            math.atan2(R[2,1], R[2,2]) * 180/math.pi,  # Roll
            math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2)) * 180/math.pi,  # Pitch
            math.atan2(R[1,0], R[0,0]) * 180/math.pi   # Yaw
        ])
        f.write(f"Rotation angles (Roll, Pitch, Yaw) [degrees]: {angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}\n")
    
    print(f"Stereo calibration saved to {filename}")

def load_calibration(filename):
    """Chargement des paramètres de calibration mono depuis un fichier txt"""
    mtx = np.zeros((3,3))
    dist = np.zeros((1,5))
    img_size = (0, 0)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Recherche des sections
    for i, line in enumerate(lines):
        if "CAMERA MATRIX" in line:
            # Lecture de la matrice 3x3
            for j in range(3):
                values = lines[i+1+j].strip().split()
                mtx[j] = [float(v) for v in values]
        
        elif "DISTORTION COEFFICIENTS:" in line:
            values = lines[i+1].strip().split()
            dist[0] = [float(v) for v in values[:5]]  # Prend les 5 premiers coefficients
        
        elif "IMAGE SIZE" in line:
            values = lines[i+1].strip().split()
            img_size = (int(values[0]), int(values[1]))
    
    return mtx, dist, img_size

def load_stereo_calibration(filename):
    """Chargement des paramètres de calibration stéréo depuis un fichier txt"""
    R = np.zeros((3,3))
    T = np.zeros((3,1))
    E = np.zeros((3,3))
    F = np.zeros((3,3))
    img_size = (0, 0)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Recherche des sections
    for i, line in enumerate(lines):
        if "ROTATION MATRIX" in line:
            for j in range(3):
                values = lines[i+1+j].strip().split()
                R[j] = [float(v) for v in values]
        
        elif "TRANSLATION VECTOR" in line:
            for j in range(3):
                T[j,0] = float(lines[i+1+j].strip())
        
        elif "ESSENTIAL MATRIX" in line:
            for j in range(3):
                values = lines[i+1+j].strip().split()
                E[j] = [float(v) for v in values]
        
        elif "FUNDAMENTAL MATRIX" in line:
            for j in range(3):
                values = lines[i+1+j].strip().split()
                F[j] = [float(v) for v in values]
        
        elif "IMAGE SIZE" in line:
            values = lines[i+1].strip().split()
            img_size = (int(values[0]), int(values[1]))
    
    return R, T, E, F, img_size

if __name__ == "__main__":
    # Calibration Caméra 1
    print("=== Calibrating Camera 1 ===")
    mtx1, dist1, img_size1 = mono_calibration(
        "Photo/camera1/*.jpg", 
        chessboard_size=(9,6), 
        square_size=25,  # en cm
        show_corners=True)
    
    print("\nCamera 1 Matrix (Intrinsic Matrix):")
    print(mtx1)
    print("\nCamera 1 Distortion Coefficients:")
    print(dist1)
    
    save_calibration("camera1_calibration.txt", mtx1, dist1, img_size1)
    
    # Calibration Caméra 2
    print("\n=== Calibrating Camera 2 ===")
    mtx2, dist2, img_size2 = mono_calibration(
        "Photo/camera2/*.jpg", 
        chessboard_size=(9,6), 
        square_size=25,  # en cm
        show_corners=True)
    
    print("\nCamera 2 Matrix (Intrinsic Matrix):")
    print(mtx2)
    print("\nCamera 2 Distortion Coefficients:")
    print(dist2)
    
    save_calibration("camera2_calibration.txt", mtx2, dist2, img_size2)
    
    # Calibration Stéréo
    print("\n=== Stereo Calibration ===")
    R, T, E, F, stereo_img_size = stereo_calibration(
        "Photo/camera1/*.jpg",
        "Photo/camera2/*.jpg",
        mtx1, dist1, mtx2, dist2,
        chessboard_size=(9,6),
        square_size=25,
        show_corners=True)
    
    save_stereo_calibration("stereo_calibration.txt", R, T, E, F, stereo_img_size)
    
    print("\n=== Calibration Complete ===")
    print("Files saved:")
    print("- camera1_calibration.txt: Camera 1 intrinsic parameters")
    print("- camera2_calibration.txt: Camera 2 intrinsic parameters") 
    print("- stereo_calibration.txt: Stereo transformation parameters")