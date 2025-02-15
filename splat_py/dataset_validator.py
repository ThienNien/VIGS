import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
class DatasetValidator:
    def __init__(self, pointcloud_ros, ros_camera):
        self.pointcloud_ros = pointcloud_ros
        self.ros_camera = ros_camera

    def check_coordinate_system(self):
        """Check for coordinate system consistency"""
        # Visualize point cloud and camera poses
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot point cloud
        ax.scatter(self.pointcloud_ros.xyz[:, 0], self.pointcloud_ros.xyz[:, 1], self.pointcloud_ros.xyz[:, 2], c='b', marker='.', s=1)
        
        # Plot camera poses
        camera_positions = self.ros_camera.pose.pose.pose.position
        ax.scatter(camera_positions.x, camera_positions.y, camera_positions.z, c='r', marker='^', s=100)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Point Cloud and Camera Poses')
        plt.show()

    def check_scale(self):
        """Check for scale consistency"""
        point_cloud_range = np.ptp(self.pointcloud_ros.xyz.numpy(), axis=0)
        camera_range = np.ptp(np.array([
            self.ros_camera.pose.pose.pose.position.x,
            self.ros_camera.pose.pose.pose.position.y,
            self.ros_camera.pose.pose.pose.position.z
        ]))
        
        print(f"Point cloud range: {point_cloud_range}")
        print(f"Camera movement range: {camera_range}")
        
        if np.any(point_cloud_range > 1000 * camera_range) or np.any(camera_range > 1000 * point_cloud_range):
            print("WARNING: There might be a scale mismatch between point cloud and camera poses.")

    def check_camera_intrinsics(self):
        """Check camera intrinsics"""
        K = self.ros_camera.roscamera[2].K
        image_shape = self.ros_camera.roscamera[1].image.shape

        print(f"Camera Intrinsics:\n{K}")
        print(f"Image shape: {image_shape}")

        # Check if principal point is roughly at the center of the image
        cx, cy = K[0, 2], K[1, 2]
        if abs(cx - image_shape[1]/2) > image_shape[1]*0.1 or abs(cy - image_shape[0]/2) > image_shape[0]*0.1:
            print("WARNING: Principal point seems to be far from the image center.")

    def check_depth_scale(self):
        """Check depth scale"""
        depth_values = self.pointcloud_ros.xyz[:, 2]  # Assuming Z is depth
        
        print(f"Depth statistics:")
        print(f"  Min: {depth_values.min()}")
        print(f"  Max: {depth_values.max()}")
        print(f"  Mean: {depth_values.mean()}")
        print(f"  Median: {torch.median(depth_values)}")
        
        if depth_values.min() < 0.1 or depth_values.max() > 1000:
            print("WARNING: Depth values seem to be in an unusual range. Check depth_scale.")

    def check_gaussian_initialization(self):
        """Check Gaussian initialization"""
        gaussians = self.pointcloud_ros.create_gaussians()
        
        print("Gaussian Initialization Statistics:")
        print(f"  Number of Gaussians: {gaussians.xyz.shape[0]}")
        print(f"  Scale - Min: {gaussians.scale.min()}, Max: {gaussians.scale.max()}, Mean: {gaussians.scale.mean()}")
        print(f"  Opacity - Min: {gaussians.opacity.min()}, Max: {gaussians.opacity.max()}, Mean: {gaussians.opacity.mean()}")
        
        # Visualize Gaussian positions and scales
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gaussians.xyz.detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(gaussians.rgb.detach().cpu().numpy() / 255.0)
        
        # Use scale as point size for visualization
        sizes = gaussians.scale.mean(dim=1).detach().cpu().numpy()
        normalized_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min())
        normalized_sizes = normalized_sizes * 0.05 + 0.01  # Scale to reasonable point sizes
        
        o3d.visualization.draw_geometries([pcd], point_show_normal=False, point_size=normalized_sizes.mean())

    def run_all_checks(self):
        print("Running all validation checks...")
        self.check_coordinate_system()
        self.check_scale()
        self.check_camera_intrinsics()
        self.check_depth_scale()
        self.check_gaussian_initialization()
        print("All checks completed.")