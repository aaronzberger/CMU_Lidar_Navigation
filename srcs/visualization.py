import numpy as np
import open3d as o3d

def pointcloud(pc, colors=None):
    pc = pc.astype('float64')
    pcd = o3d.geometry.PointCloud()

    if colors is not None:
        colors = colors.astype('float64')
        pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.points = o3d.utility.Vector3dVector(pc)

    return pcd


def vertical_lines(pts, z1, z2):
    z1 = np.full((pts.shape[0], 1), z1)
    z2 = np.full((pts.shape[0], 1), z2)

    lines1 = np.concatenate((pts, z1), axis=1)
    lines2 = np.concatenate((pts, z2), axis=1)

    corresp = [(n, n) for n in range(len(lines1))]

    lines =\
    o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            pointcloud(lines1),
            pointcloud(lines2), 
            corresp
    )

    return lines


def horizontal_lines(pts1, pts2, z):
    zeros = np.full((pts1.shape[0], 1), z)
    lines1 = np.concatenate((pts1, zeros), axis=1)
    lines2 = np.concatenate((pts2, zeros), axis=1)

    corresp = [(n, n) for n in range(len(lines1))]

    lines =\
    o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            pointcloud(lines1),
            pointcloud(lines2), 
            corresp
    )

    return lines


