import rospy
import os
import cv2
import torch
import time
from threading import Thread, Lock
import message_filters
import image_transport
from sensor_msgs.msg import PointCloud2
import splat_py.optimizer_manager
from nav_msgs.msg import Odometry
from splat_py.structs import Gaussians, Image, Camera
from splat_py.utils_ros import *
import sensor_msgs.point_cloud2 as pc2
from splat_py.config import SplatConfigs
from splat_py.trainer_modify import SplatTrainer
import random
from torchmetrics.image import StructuralSimilarityIndexMeasure

from splat_py.optimizer_manager import OptimizerManager
from splat_py.structs import GSMetrics
from splat_py.rasterize import rasterize
from splat_py.utils import (
    inverse_sigmoid,
    quaternion_to_rotation_torch,
)
import splat_py.config as cf
import tyro
import yaml
device= torch.device("cuda:0")
config = tyro.cli(SplatConfigs)
if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)
# save a copy of the config
yaml.dump(config, open(os.path.join(config.output_dir, "config.yaml"), "w"))
cam_index=0
class Subdata:
    def __init__(self):
          self.image=None
          self.depth_img=None
          self.odometry=None
          self.PCL_ros=None
          self.cam_index=0
          self.trainer=None
          image_transport.Subscriber("/zed2/zed_node/depth/depth_registered",self.depth_callback)
          image_transport.Subscriber("/zed2/zed_node/left_raw/image_raw_color",self.rgb_callback)
          rospy.Subscriber("/lvi_sam/lidar/mapping/odometry",Odometry,self.pose_callback)
    def rgb_callback(self,image):
         time.sleep(0.001)
         self.image=image
         self.Cam = RosCamera(image=self.image,pose=self.odometry,downsample_factor=config.downsample_factor,\
                    cam_idx=cam_index, device=device, config=config)
         
    def depth_callback(self,depth_image):
        time.sleep(0.001)
        self.depth_img=depth_image
        PCL_ros=Pointcloud_ros(depth_image=self.depth_img,
                    image=self.image,
                    device=device,
                    downsample_factor=config.downsample_factor,
                    config=config)
        new_gaussians = PCL_ros.create_gaussians()
            # => Gaussians exist
        new_gaussians.xyz = torch.nn.Parameter(new_gaussians.xyz)
        new_gaussians.quaternion = torch.nn.Parameter(new_gaussians.quaternion)
        new_gaussians.scale = torch.nn.Parameter(new_gaussians.scale)
        new_gaussians.opacity = torch.nn.Parameter(new_gaussians.opacity)
        new_gaussians.rgb = torch.nn.Parameter(new_gaussians.rgb)
        print("a")
        if self.cam_index == 0:
            cf.mapping_kf.append([self.Cam,new_gaussians])
    #     # trainer = SplatTrainer(mapping_kf,config= config)
            self.trainer = SplatTrainer(cf.mapping_kf[0],config= config)
            newCam,newGaussians= cf.mapping_kf[0]
            self.trainer.train(self.cam_index,newCam)
        if self.cam_index%5 == 0 and self.cam_index!=0:
                cf.mapping_kf.append([self.Cam,new_gaussians])
                cf.is_newKF=True
                
                newCam,newGaussians= cf.mapping_kf[-1]
                # trainer = SplatTrainer(cf.mapping_kf[-1],config= config)
                self.trainer.densify_postfix(new_gaussians=newGaussians, clone_mask=np.int64(len(newGaussians.xyz)))
                print(f"NO of gaussian map {len(self.trainer.gaussians.xyz)}")
                t=time.time()
                self.trainer.train(self.cam_index,newCam)
                print("FPS test", 1/(time.time()-t))

        print(f"Success ! {self.cam_index}")
        self.cam_index+=1
    def pose_callback(self,odometry):
         time.sleep(0.001)
         self.odometry=odometry
          