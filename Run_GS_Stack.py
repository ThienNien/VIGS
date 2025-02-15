# refactored_code.py

import rospy
import os
import cv2
import torch
import time
import random
import struct
import torch.multiprocessing as mp
from threading import Thread, Lock
import message_filters
import image_transport
from sensor_msgs.msg import Image , PointCloud2
from nav_msgs.msg import Odometry
from splat_py.structs import Gaussians, Camera
from splat_py.utils_ros import RosCamera, Pointcloud_ros 
import sensor_msgs.point_cloud2 as pc2
from splat_py.config import SplatConfigs
from splat_py.trainer_modify import SplatTrainer
from torchmetrics.image import StructuralSimilarityIndexMeasure
from splat_py.sub_data import Subdata
from splat_py.optimizer_manager import OptimizerManager
from splat_py.structs import GSMetrics
from splat_py.rasterize import rasterize
from splat_py.utils import inverse_sigmoid, quaternion_to_rotation_torch
import config as cf
import tyro
import yaml
import rerun as rr
import threading
import numpy as np
from sensor_msgs.msg import PointField
import open3d as o3d
# Global variables
device = torch.device("cuda:0")
config = tyro.cli(SplatConfigs)
_global_vars = {
    'gaussian_visualizer': None,
    'cam_index': 0,
    'stack_gaussian': None,
    'trainer': None
}


def setup_output_dir():
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    yaml.dump(config, open(os.path.join(config.output_dir, "config.yaml"), "w"))


def initialize_globals():
    _global_vars['cam_index'] = 0
    _global_vars['stack_gaussian'] = None
    cf.mapping_kf = []
    cf.new_gaussians = None
    cf.is_newKF = False
    cf.using_sync = True
    cf.window_size = 5
    cf.window_step = cf.window_size - 2
    cf.window_buffer = {
        'images': [],
        'depths': [],
        'poses': []
    }
    cf.frames_since_last_window = 0
    
    # try:
    #     _global_vars['gaussian_visualizer'] = GaussianSplatVisualizer()
    #     rospy.loginfo("Gaussian Splat visualizer initialized successfully")
    # except Exception as e:
    #     rospy.logerr(f"Failed to initialize Gaussian Splat visualizer: {e}")
    #     _global_vars['gaussian_visualizer'] = None

def combine_and_downsample_points(points_list, colors_list, middle_points, middle_colors, voxel_size=0.1):
    """
    Combine points from all frames and apply voxel grid downsampling,
    ensuring middle frame points are preserved.
    """
    # try:
    if not points_list or not middle_points:
        rospy.logwarn("Empty point lists received")
        return None, None

    # Convert lists to numpy arrays if they aren't already
    middle_points = np.array(middle_points)
    middle_colors = np.array(middle_colors)

    # Create Open3D point cloud for middle frame
    middle_pcd = o3d.geometry.PointCloud()
    middle_pcd.points = o3d.utility.Vector3dVector(middle_points)
    middle_pcd.colors = o3d.utility.Vector3dVector(middle_colors)

    # Collect points from other frames
    other_points = []
    other_colors = []
    middle_idx = len(points_list) // 2

    for i in range(len(points_list)):
        if i != middle_idx:  # Skip middle frame as it's handled separately
            other_points.extend(points_list[i])
            other_colors.extend(colors_list[i])

    if other_points:
        # Create Open3D point cloud for other frames
        other_pcd = o3d.geometry.PointCloud()
        other_pcd.points = o3d.utility.Vector3dVector(np.array(other_points))
        other_pcd.colors = o3d.utility.Vector3dVector(np.array(other_colors))

        # Downsample other points
        other_pcd_downsampled = other_pcd.voxel_down_sample(voxel_size=voxel_size)

        if other_pcd_downsampled is None or len(other_pcd_downsampled.points) == 0:
            rospy.logwarn("Downsampling resulted in empty point cloud")
            return np.asarray(middle_pcd.points), np.asarray(middle_pcd.colors)

        # Combine downsampled other points with middle frame points
        combined_points = np.vstack([
            np.asarray(middle_pcd.points),
            np.asarray(other_pcd_downsampled.points)
        ])
        combined_colors = np.vstack([
            np.asarray(middle_pcd.colors),
            np.asarray(other_pcd_downsampled.colors)
        ])
    else:
        # If no other points, just use middle frame points
        combined_points = np.asarray(middle_pcd.points)
        combined_colors = np.asarray(middle_pcd.colors)

    rospy.loginfo(f"Combined point cloud size: {len(combined_points)}")
    return combined_points, combined_colors

    # except Exception as e:
    #     rospy.logerr(f"Error in combine_and_downsample_points: {e}")
    #     return None, None

def create_point_cloud2(points, colors, header):
    """Create a PointCloud2 message from points and colors."""
    try:
        # Combine points and colors into structured array
        cloud_data = np.zeros(len(points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.float32)
        ])
        
        cloud_data['x'] = points[:, 0]
        cloud_data['y'] = points[:, 1]
        cloud_data['z'] = points[:, 2]
        
        # Pack RGB values into float32
        rgb_packed = np.zeros(len(colors), dtype=np.float32)
        for i, color in enumerate(colors):
            rgb_packed[i] = struct.unpack('f', struct.pack('I', 
                int(color[0]*255) << 16 | int(color[1]*255) << 8 | int(color[2]*255)))[0]
        cloud_data['rgb'] = rgb_packed
        
        # Create and return PointCloud2 message
        return pc2.create_cloud(header,
            [PointField('x', 0, PointField.FLOAT32, 1),
             PointField('y', 4, PointField.FLOAT32, 1),
             PointField('z', 8, PointField.FLOAT32, 1),
             PointField('rgb', 12, PointField.FLOAT32, 1)],
            cloud_data)
    except Exception as e:
        rospy.logerr(f"Error in create_point_cloud2: {e}")
        return None

def process_window_data():
    """Process data in the sliding window to create a combined point cloud."""
    try:
        if len(cf.window_buffer['images']) < cf.window_size:
            rospy.logwarn(f"Window buffer not full yet: {len(cf.window_buffer['images'])}/{cf.window_size}")
            return None
        
        middle_idx = cf.window_size // 2
        middle_frame = {
            'image': cf.window_buffer['images'][middle_idx],
            'pose': cf.window_buffer['poses'][middle_idx]
        }
        
        # Process points frame by frame
        points_list = []
        colors_list = []
        
        for i in range(cf.window_size):
            current_points = []
            current_colors = []
            
            try:
                pc_data = cf.window_buffer['depths'][i]
                gen = pc2.read_points(pc_data, field_names=("x", "y", "z", "rgb"), skip_nans=True)
                
                for point in gen:
                    try:
                        x, y, z, rgb = point
                        rgb_val = struct.unpack('I', struct.pack('f', rgb))[0]
                        r = ((rgb_val >> 16) & 0xFF) / 255.0
                        g = ((rgb_val >> 8) & 0xFF) / 255.0
                        b = (rgb_val & 0xFF) / 255.0
                        
                        current_points.append([x, y, z])
                        current_colors.append([r, g, b])
                    except Exception as e:
                        rospy.logwarn(f"Error processing point: {e}")
                        continue
                        
                points_list.append(current_points)
                colors_list.append(current_colors)
                
            except Exception as e:
                rospy.logwarn(f"Error processing frame {i}: {e}")
                continue
        
        if not points_list:
            rospy.logwarn("No valid points found in window")
            return None

        # Use voxel_size from config if available, otherwise use default
        voxel_size = getattr(cf, 'voxel_size', 0.05)
        
        # Combine and downsample points while preserving middle frame
        combined_points, combined_colors = combine_and_downsample_points(
            points_list, colors_list,
            points_list[middle_idx], colors_list[middle_idx],
            voxel_size=voxel_size
        )
        
        if combined_points is None:
            return None
            
        # Convert to PointCloud2 message
        combined_cloud = create_point_cloud2(
            points=combined_points,
            colors=combined_colors,
            header=cf.window_buffer['depths'][middle_idx].header
        )
        
        if combined_cloud is None:
            rospy.logwarn("Failed to create combined point cloud message")
            return None
        
        return [middle_frame['image'], combined_cloud, middle_frame['pose']]
        
    except Exception as e:
        rospy.logerr(f"Error in process_window_data: {e}")
        return None


def _callback(depth_image, image, odometry):
    try:
        # Add new data to window buffer
        cf.window_buffer['images'].append(image)
        cf.window_buffer['depths'].append(depth_image)
        cf.window_buffer['poses'].append(odometry)
        cf.frames_since_last_window += 1
        
        # Process window when enough frames are collected and we've moved enough steps
        if (len(cf.window_buffer['images']) >= cf.window_size and 
            cf.frames_since_last_window >= cf.window_step):
            
            # Process window data
            processed_data = process_window_data()
            if processed_data:
                cf.mapping_kf.append(processed_data)
                if not hasattr(cf, 'new_KF'):
                    cf.new_KF = []
                cf.new_KF.append(len(cf.mapping_kf) - 1)
                # rospy.loginfo(f"New KF added from window {_global_vars['cam_index']}")
            
            # Slide window by removing steps number of oldest frames
            # Keep one frame for overlap
            for _ in range(cf.window_step):
                if cf.window_buffer['images']:
                    cf.window_buffer['images'].pop(0)
                    cf.window_buffer['depths'].pop(0)
                    cf.window_buffer['poses'].pop(0)
            
            # Reset the frame counter
            cf.frames_since_last_window = 0
        
        # Prevent buffer from growing too large
        max_buffer_size = cf.window_size + 2  # Allow slight overflow
        if len(cf.window_buffer['images']) > max_buffer_size:
            cf.window_buffer['images'] = cf.window_buffer['images'][-cf.window_size:]
            cf.window_buffer['depths'] = cf.window_buffer['depths'][-cf.window_size:]
            cf.window_buffer['poses'] = cf.window_buffer['poses'][-cf.window_size:]
        
        _global_vars['cam_index'] += 1
        # rospy.loginfo(f"Success! Camera index: {_global_vars['cam_index']}")
        
    except Exception as e:
        rospy.logerr(f"Error in _callback: {e}")

def parameterize(KF):
    Cam = RosCamera(
        image=KF[0], 
        pose=KF[2], 
        downsample_factor=config.downsample_factor, 
        cam_idx=_global_vars['cam_index'], 
        device=device, 
        config=config
    )
    PCL_ros = Pointcloud_ros(
        depth_image=KF[1], 
        image=KF[0],
        pose=KF[2],
        device=device, 
        downsample_factor=config.downsample_factor, 
        config=config
    )
    new_gaussians = PCL_ros.create_gaussians()
    new_gaussians.xyz = torch.nn.Parameter(new_gaussians.xyz)
    new_gaussians.quaternion = torch.nn.Parameter(new_gaussians.quaternion)
    new_gaussians.scale = torch.nn.Parameter(new_gaussians.scale)
    new_gaussians.opacity = torch.nn.Parameter(new_gaussians.opacity)
    new_gaussians.rgb = torch.nn.Parameter(new_gaussians.rgb)
    return Cam, new_gaussians

def mapping():
    """Main mapping function that processes keyframes and trains the Gaussian model."""
    Cam_KF = []

    # Wait for first keyframe
    while not cf.mapping_kf:
        time.sleep(1e-15)
    
    # Initialize with first keyframe
    try:
        newCam, newGaussians = parameterize(cf.mapping_kf[0])
        if newGaussians is not None:
            _global_vars['trainer'] = SplatTrainer([newCam, newGaussians], config=config)
            Cam_KF.append(newCam)
            _global_vars['trainer'].train(0, Cam_KF, True)
            
    #         if _global_vars['gaussian_visualizer']:
    #             _global_vars['gaussian_visualizer'].update_gaussian_points(newGaussians)
    except Exception as e:
        rospy.logerr(f"Error in initial training: {e}")

    # Main mapping loop
    i = 1
    while True:
        try:
            if cf.mapping_kf:
                time_render = time.time()
                
                if cf.new_KF:
                    train_idx = cf.new_KF.pop(0)
                    newCam, newGaussians = parameterize(cf.mapping_kf[train_idx])
                    
                    if newGaussians is not None and _global_vars['trainer'] is not None:
                        _global_vars['trainer'].densify_postfix(new_gaussians=newGaussians)
                        
                        rospy.loginfo(f"Added Gaussian: There are {len(newGaussians.xyz)} has been added")
                        Cam_KF.append(newCam)
                        cf.is_newKF = True
                        
                        # if _global_vars['gaussian_visualizer']:
                        #     _global_vars['gaussian_visualizer'].update_gaussian_points(newGaussians)
                else:
                    cf.is_newKF = False

                if _global_vars['trainer'] is not None:
                    _global_vars['trainer'].train(i, Cam_KF, cf.is_newKF)
                
                i += 1
                # fps = 1 / (time.time() - time_render)
                # rospy.loginfo(f"Render FPS: {fps:.2f}")
            
            if len(cf.mapping_kf) > 1000:
                break
                
            time.sleep(0.01)
            
        except Exception as e:
            rospy.logerr(f"Error in mapping iteration {i}: {e}")
            time.sleep(0.1)

    rospy.loginfo("Mapping completed")
    
def listener():
    rospy.init_node('GS_Map')
    if cf.using_sync:
        subscribers = [
            message_filters.Subscriber("/vins_estimator/point_cloud_2",PointCloud2),
            message_filters.Subscriber("/vins_estimator/keyframe_cam",Image),
            message_filters.Subscriber("/vins_estimator/keyframe_pose",Odometry)
        ]
        mf = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=10, slop=0.1)
        mf.registerCallback(_callback)

if __name__ == '__main__':
    setup_output_dir()
    initialize_globals()

    mapping_thread = Thread(target=mapping)
    mapping_thread.start()

    listener()
