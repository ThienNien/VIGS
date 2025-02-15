import rospy
import os
import cv2
import torch
import time
import struct
import torch.multiprocessing as mp
from threading import Thread, Lock
import message_filters
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from splat_py.structs import Gaussians, Camera
from splat_py.utils_ros import RosCamera, Pointcloud_ros
import sensor_msgs.point_cloud2 as pc2
from splat_py.config import SplatConfigs
from splat_py.trainer_modify import SplatTrainer
import open3d as o3d
import numpy as np
import tyro
import yaml

class GaussianSplatMapper:
    def __init__(self):
        """Initialize the Gaussian Splatting Mapping system."""
        self.device = torch.device("cuda:0")
        self.config = tyro.cli(SplatConfigs)
        self.setup_parameters()
        self.initialize_ros()
        
    def setup_parameters(self):
        """Set up system parameters and create output directory."""
        # System state
        self.cam_index = 0
        self.trainer = None
        self.stack_gaussian = None
        
        # Mapping parameters
        self.window_size = 15
        self.window_step = self.window_size - 1
        self.voxel_size = 0.03 #0.05
        
        # Data containers
        self.mapping_kf = []
        self.new_KF = []
        self.window_buffer = {
            'images': [],
            'depths': [],
            'poses': []
        }
        self.frames_since_last_window = 0
        
        # Create output directory
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        yaml.dump(self.config, open(os.path.join(self.config.output_dir, "config.yaml"), "w"))

    def initialize_ros(self):
        """Initialize ROS node and subscribers."""
        rospy.init_node('gaussian_splat_mapper')
        subscribers = [
            message_filters.Subscriber("/vins_estimator/point_cloud_2", PointCloud2),
            message_filters.Subscriber("/vins_estimator/keyframe_cam", Image),
            message_filters.Subscriber("/vins_estimator/keyframe_pose", Odometry)
        ]
        sync = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=10, slop=0.1)
        sync.registerCallback(self.callback)

    def callback(self, depth_image, image, odometry):
        # t = time.time()
        """Process incoming synchronized ROS messages."""
        try:
            if self.cam_index == 0:
                # Process first frame immediately for initialization
                self.process_first_frame(image, depth_image, odometry)
            else:
                # Add to window buffer
                self.window_buffer['images'].append(image)
                self.window_buffer['depths'].append(depth_image)
                self.window_buffer['poses'].append(odometry)
                self.frames_since_last_window += 1
                
                # Process window if ready
                self.process_window_if_ready()
                
                # Maintain buffer size
                self.maintain_buffer_size()
            
            self.cam_index += 1
            
        except Exception as e:
            rospy.logerr(f"Error in callback: {str(e)}")
        # print(" FPS of Callback",1/(time.time()-t))

    def process_first_frame(self, image, depth_image, odometry):
        """Process the first frame for immediate initialization."""
        self.mapping_kf.append([image, depth_image, odometry])
        self.new_KF.append(len(self.mapping_kf) - 1)
        
        # Initialize window buffer
        self.window_buffer['images'].append(image)
        self.window_buffer['depths'].append(depth_image)
        self.window_buffer['poses'].append(odometry)
        self.frames_since_last_window = 1
        
        rospy.loginfo("First frame processed for initialization")

    def process_window_if_ready(self):
        """Process the window if enough frames have been collected."""
        if (len(self.window_buffer['images']) >= self.window_size and 
            self.frames_since_last_window >= self.window_step):
            t = time.time()
            processed_data = self.process_window_data()
            print("FPS of process_window_data",1/(time.time()-t))

            if processed_data:
                self.mapping_kf.append(processed_data)
                self.new_KF.append(len(self.mapping_kf) - 1)
            
            # Slide window
            for _ in range(self.window_step):
                if self.window_buffer['images']:
                    for key in self.window_buffer:
                        self.window_buffer[key].pop(0)
            
            self.frames_since_last_window = 0

    def maintain_buffer_size(self):
        """Ensure buffer doesn't grow too large."""
        max_buffer_size = self.window_size + 2
        if len(self.window_buffer['images']) > max_buffer_size:
            for key in self.window_buffer:
                self.window_buffer[key] = self.window_buffer[key][-self.window_size:]

    def process_window_data(self):
        """Process window data to create combined point cloud."""
        try:
            if len(self.window_buffer['images']) < self.window_size:
                return None
            
            middle_idx = self.window_size // 2
            middle_frame = {
                'image': self.window_buffer['images'][middle_idx],
                'pose': self.window_buffer['poses'][middle_idx]
            }
            
            # Process points for each frame
            points_list = []
            colors_list = []
            
            for depth_data in self.window_buffer['depths']:
                current_points, current_colors = self.extract_points_from_cloud(depth_data)
                if current_points:
                    points_list.append(current_points)
                    colors_list.append(current_colors)
            
            if not points_list:
                return None

            # Combine and downsample points
            combined_points, combined_colors = self.combine_points(
                points_list, 
                colors_list,
                points_list[middle_idx], 
                colors_list[middle_idx]
            )
            
            if combined_points is None:
                return None
                
            # Create point cloud message
            combined_cloud = self.create_point_cloud_msg(
                points=combined_points,
                colors=combined_colors,
                header=self.window_buffer['depths'][middle_idx].header
            )
            
            return [middle_frame['image'], combined_cloud, middle_frame['pose']]
            
        except Exception as e:
            rospy.logerr(f"Error processing window data: {str(e)}")
            return None


    def extract_points_from_cloud(self, cloud_msg):
        """Extract points and colors from a PointCloud2 message."""
        try:
            points = []
            colors = []
            
            gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
            for point in gen:
                x, y, z, rgb = point
                rgb_val = struct.unpack('I', struct.pack('f', rgb))[0]
                r = ((rgb_val >> 16) & 0xFF) / 255.0
                g = ((rgb_val >> 8) & 0xFF) / 255.0
                b = (rgb_val & 0xFF) / 255.0
                
                points.append([x, y, z])
                colors.append([r, g, b])
                
            return points, colors
        except Exception as e:
            rospy.logwarn(f"Error extracting points: {str(e)}")
            return None, None

    def combine_points(self, points_list, colors_list, middle_points, middle_colors):
        """Combine and downsample points while preserving middle frame."""
        try:
            # Create Open3D point cloud for middle frame
            middle_pcd = o3d.geometry.PointCloud()
            middle_pcd.points = o3d.utility.Vector3dVector(np.array(middle_points))
            middle_pcd.colors = o3d.utility.Vector3dVector(np.array(middle_colors))

            # Process other frames
            other_points = []
            other_colors = []
            middle_idx = len(points_list) // 2

            for i in range(len(points_list)):
                if i != middle_idx:
                    other_points.extend(points_list[i])
                    other_colors.extend(colors_list[i])

            if other_points:
                # Create and downsample other points
                other_pcd = o3d.geometry.PointCloud()
                other_pcd.points = o3d.utility.Vector3dVector(np.array(other_points))
                other_pcd.colors = o3d.utility.Vector3dVector(np.array(other_colors))
                
                other_pcd_downsampled = other_pcd.voxel_down_sample(voxel_size=self.voxel_size)
                
                # Combine points
                combined_points = np.vstack([
                    np.asarray(middle_pcd.points),
                    np.asarray(other_pcd_downsampled.points)
                ])
                combined_colors = np.vstack([
                    np.asarray(middle_pcd.colors),
                    np.asarray(other_pcd_downsampled.colors)
                ])
            else:
                combined_points = np.asarray(middle_pcd.points)
                combined_colors = np.asarray(middle_pcd.colors)

            return combined_points, combined_colors
            
        except Exception as e:
            rospy.logerr(f"Error combining points: {str(e)}")
            return None, None

    def create_point_cloud_msg(self, points, colors, header):
        """Create a PointCloud2 message from points and colors."""
        try:
            cloud_data = np.zeros(len(points), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.float32)
            ])
            
            cloud_data['x'] = points[:, 0]
            cloud_data['y'] = points[:, 1]
            cloud_data['z'] = points[:, 2]
            
            # Pack RGB values
            rgb_packed = np.zeros(len(colors), dtype=np.float32)
            for i, color in enumerate(colors):
                rgb_packed[i] = struct.unpack('f', struct.pack('I', 
                    int(color[0]*255) << 16 | int(color[1]*255) << 8 | int(color[2]*255)))[0]
            cloud_data['rgb'] = rgb_packed
            
            return pc2.create_cloud(header,
                [PointField('x', 0, PointField.FLOAT32, 1),
                 PointField('y', 4, PointField.FLOAT32, 1),
                 PointField('z', 8, PointField.FLOAT32, 1),
                 PointField('rgb', 12, PointField.FLOAT32, 1)],
                cloud_data)
        except Exception as e:
            rospy.logerr(f"Error creating point cloud message: {str(e)}")
            return None

    def parameterize(self, keyframe):
        """Create camera and Gaussian parameters from keyframe data."""
        try:
            # Create ROS camera object
            camera = RosCamera(
                image=keyframe[0],
                pose=keyframe[2],
                downsample_factor=self.config.downsample_factor,
                cam_idx=self.cam_index,
                device=self.device,
                config=self.config
            )
            
            # Create point cloud object
            pcl_ros = Pointcloud_ros(
                depth_image=keyframe[1],
                image=keyframe[0],
                pose=keyframe[2],
                device=self.device,
                downsample_factor=self.config.downsample_factor,
                config=self.config
            )
            
            # Create and parameterize Gaussians
            gaussians = pcl_ros.create_gaussians()
            if gaussians is not None:
                gaussians.xyz = torch.nn.Parameter(gaussians.xyz)
                gaussians.quaternion = torch.nn.Parameter(gaussians.quaternion)
                gaussians.scale = torch.nn.Parameter(gaussians.scale)
                gaussians.opacity = torch.nn.Parameter(gaussians.opacity)
                gaussians.rgb = torch.nn.Parameter(gaussians.rgb)
                
            return camera, gaussians
            
        except Exception as e:
            rospy.logerr(f"Error in parameterization: {str(e)}")
            return None, None

    def mapping_loop(self):
        """Main mapping loop that processes keyframes and trains the model."""
        camera_keyframes = []
        
        # Wait for first keyframe
        while not self.mapping_kf:
            time.sleep(1e-15)
        
        # Initialize with first keyframe
        try:
            camera, gaussians = self.parameterize(self.mapping_kf[0])
            if gaussians is not None:
                self.trainer = SplatTrainer([camera, gaussians], config=self.config)
                camera_keyframes.append(camera)
                self.trainer.train(0, camera_keyframes, True)
        except Exception as e:
            rospy.logerr(f"Error in initial training: {str(e)}")

        # Main processing loop
        iteration = 1
        while not rospy.is_shutdown():
            try:
                if self.mapping_kf:
                    if self.new_KF:
                        train_idx = self.new_KF.pop(0)
                        camera, gaussians = self.parameterize(self.mapping_kf[train_idx])
                        
                        if gaussians is not None and self.trainer is not None:
                            self.trainer.densify_postfix(new_gaussians=gaussians)
                            camera_keyframes.append(camera)
                            rospy.loginfo(f"Added {len(gaussians.xyz)} new Gaussians")
                    
                    if self.trainer is not None:
                        self.trainer.train(iteration, camera_keyframes, bool(self.new_KF))
                    
                    iteration += 1
                
                if len(self.mapping_kf) > 1000:  # Safety limit
                    break
                    
                time.sleep(0.01)
                
            except Exception as e:
                rospy.logerr(f"Error in mapping iteration {iteration}: {str(e)}")
                time.sleep(0.1)

        rospy.loginfo("Mapping completed")

def main():
    try:
        mapper = GaussianSplatMapper()
        mapping_thread = Thread(target=mapper.mapping_loop)
        mapping_thread.start()
        rospy.spin()
    except Exception as e:
        rospy.logerr(f"Error in main: {str(e)}")
    finally:
        rospy.signal_shutdown("Mapping completed")

if __name__ == '__main__':
    main()