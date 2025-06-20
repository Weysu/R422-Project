import numpy as np
import open3d as o3d

# 1. Charger les points 3D
points = np.loadtxt('reconstruction_3d_corrected.txt', skiprows=1)

# 2. Créer un nuage de points Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# (Optionnel) Estimer les normales pour le mesh
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=50.0, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(100)

# 3. Reconstruire la surface (Poisson Reconstruction)
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=8)

# (Optionnel) Nettoyer le mesh avec un seuil de densité
# Supprimer les triangles avec densité faible
densities = np.asarray(densities)
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

# 4. Affichage du résultat
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# 5. Sauvegarde en fichier .ply
o3d.io.write_triangle_mesh("mesh_reconstruit.ply", mesh)
print("Mesh sauvegardé dans 'mesh_reconstruit.ply'")
