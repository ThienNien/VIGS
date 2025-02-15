import time
import math
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from sensor_msgs.msg import PointCloud, PointField
import struct
import open3d as o3d
# from optimizer_manager import OptimizerManager
import tf
import torch
import cv2
from tf.transformations import euler_from_quaternion
from math import degrees
import numpy as np
from cv_bridge import CvBridge
from splat_py.structs import Gaussians, Image, Camera
from splat_py.config import SplatConfig
import torch.nn as nn
import copy
import tf

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import sensor_msgs.point_cloud2 as pc2
from splat_py.utils import inverse_sigmoid, compute_initial_scale_from_sparse_points, compute_initial_scale_from_sparse_points_D

from splat_py.read_colmap import (
    read_images_binary,
    read_points3D_binary,
    read_cameras_binary,
    qvec2rotmat,
)
from scipy.spatial.transform import Rotation

device= torch.device("cuda")
bridge = CvBridge()

class GaussianSplattingDataset:
    """
    Generic Gaussian Splatting Dataset class

    Classes that inherit from this class should have the following variables:

    device: torch device
    xyz: Nx3 tensor of points
    rgb: Nx3 tensor of rgb values

    images: list of Image objects
    cameras: dict of Camera objects

    """

    def __init__(self, config):
        self.config = config

    def verify_loaded_points(self):
        """
        Verify that the values loaded from the dataset are consistent
        """
        N = self.xyz.shape[0]
        assert self.xyz.shape[1] == 3
        assert self.rgb.shape[0] == N
        assert self.rgb.shape[1] == 3

    def create_gaussians(self):
        """
        Create gaussians object from the dataset
        """
        self.verify_loaded_points()
        N = self.xyz.shape[0]

        initial_opacity = torch.ones(N, 1) * inverse_sigmoid(self.config.initial_opacity)
        # compute scale based on the density of the points around each point

        initial_scale = compute_initial_scale_from_sparse_points(
            self.xyz,
            num_neighbors=self.config.initial_scale_num_neighbors,
            neighbor_dist_to_scale_factor=self.config.initial_scale_factor,
            max_initial_scale=self.config.max_initial_scale,
        )
        # initial_scale =torch.FloatTensor(N, 3).uniform_(2, 4)
        
        # initial_scale=torch.dot(initial_scale,5.0)

        initial_quaternion = torch.zeros(N, 4)
        initial_quaternion[:, 0] = 1.0

        return Gaussians(
            xyz=self.xyz.to(self.device),
            rgb=self.rgb.to(self.device),
            opacity=initial_opacity.to(self.device),
            scale=initial_scale.to(self.device),
            quaternion=initial_quaternion.to(self.device),
        )

class RosCamera(GaussianSplattingDataset):
    def __init__(
        self,
        image,
        pose,
        cam_idx,
        device: torch.device,
        downsample_factor: int,
        config: SplatConfig,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.device = device
        self.downsample_factor = downsample_factor
        self.pose=pose
        #pose here
        self.qvec=None
        self.tvec=None
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        camera_T_world = torch.eye(4)
        camera_T_world = torch.tensor(self.pose_to_mat(pose),dtype=torch.float32)

        K = torch.tensor(config.K, dtype = torch.float32,device=device)
        self.roscamera =  [cam_idx, 
                        Image(image=torch.from_numpy(cv_image).to(torch.uint8).to(device),camera_T_world=camera_T_world.to(self.device),), 
                        Camera(width=cv_image.shape[1], height=cv_image.shape[0],K = K)]
        # self.roscamera[cam_inx, Image(), camera()]

        #I think it need distortion / check tomorrow
    
    def pose_to_mat(self,pose):
        # Create rotation matrix from quaternion
        x = pose.pose.pose.position.x
        y = pose.pose.pose.position.y
        z = pose.pose.pose.position.z

        qx = pose.pose.pose.orientation.x
        qy = pose.pose.pose.orientation.y
        qz = pose.pose.pose.orientation.z
        qw = pose.pose.pose.orientation.w

        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        
        # Create translation vector
        t = np.array([x, y, z])
        
        # Combine into 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        R2 = self.config.R_calib.T
        T2 = -self.config.R_calib @ self.config.T_calib

        Tc = np.eye(4)
        Tc[:3, :3] =R2
        Tc[:3, 3] = T2
        
        return Tc @ np.linalg.inv(T)

class Pointcloud_ros(GaussianSplattingDataset):
    def __init__(self, image, depth_image, pose, device: torch.device, 
                 downsample_factor: int, config: SplatConfig) -> None:
        super().__init__(config)
        self.config = config
        self.device = device
        self.depth_scale = 1
        self.downsample_factor = downsample_factor
        using_camdepth = 0

        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Read pointcloud data
        gen = pc2.read_points(depth_image, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        xyz = []
        rgb_ = []
        for i, point in enumerate(gen):
            x, y, z, rgb = point
            
            # Convert RGB from float to (r, g, b) tuple
            rgb = struct.unpack('I', struct.pack('f', rgb))[0]
            r = ((rgb >> 16) & 0xFF)
            g = (rgb >> 8) & 0xFF
            b = rgb & 0xFF
            
            # Normalize RGB to [0,1] range
            rgb_.append([r/255.0, g/255.0, b/255.0])
            xyz.append([x,y,z])
        
        # Convert to tensor and ensure correct dtype
        self.xyz = torch.tensor(xyz, dtype=torch.float32)
        self.rgb = torch.tensor(rgb_, dtype=torch.float32)

        # Debug print
        # print(f"RGB tensor stats - min: {self.rgb.min()}, max: {self.rgb.max()}, mean: {self.rgb.mean()}")
    
    def add_Gaussian_map(self, gaussians,create_gaussian):
        gaussians.xyz = torch.nn.Parameter(gaussians.xyz)
        gaussians.quaternion = torch.nn.Parameter(gaussians.quaternion)
        gaussians.scale = torch.nn.Parameter(gaussians.scale)
        gaussians.opacity = torch.nn.Parameter(gaussians.opacity)
        gaussians.rgb = torch.nn.Parameter(gaussians.rgb)
        if create_gaussian:
            self.keyframe_idx = torch.ones((gaussians.xyz.shape[0],1), dtype=torch.bool, device="cuda")
            # print(self.keyframe_idx.shape[1],"self.keyframe_idx.shape[1]")
        else:
            
            new_keyframe_idx = torch.zeros((gaussians.xyz.shape[0], self.keyframe_idx.shape[1]), device="cuda", dtype=torch.bool)
            self.keyframe_idx = torch.concat([ self.keyframe_idx,
                                                new_keyframe_idx], dim=0)
            
    