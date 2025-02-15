# refactored_code.py

import rospy
import os
import cv2
import torch
import time
import random
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

device = torch.device("cuda:0")
config = tyro.cli(SplatConfigs)



def setup_output_dir():
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    yaml.dump(config, open(os.path.join(config.output_dir, "config.yaml"), "w"))

def initialize_globals():
    global cam_index, stack_gaussian, trainer
    cam_index = 0
    stack_gaussian = None
    cf.mapping_kf = []
    cf.new_gaussians = None
    cf.is_newKF = False
    cf.using_sync = True
    cf.depth_image = None
    cf.img = None
    cf.odometry = None
    cf.new_KF = []
    cf.using_camdepth=True
    trainer = None

def _callback(depth_image, image, odometry):
    global cam_index, trainer
    cf.depth_image = depth_image
    cf.img = image
    cf.odometry = odometry

    if all([cf.depth_image, cf.img, cf.odometry]):
        if cam_index == 0 or (cam_index % 2 == 0 and cam_index > 0):
            cf.mapping_kf.append([image, depth_image, odometry])
            cf.new_KF.append(len(cf.mapping_kf) - 1)
            
            print("New KF added")
        cam_index += 1
    print(f"Success! {cam_index}")

def visualize_uv_on_mat(cam, gaussian):
    # Create a blank black image
    image = cam.roscamera[1].image.detach().cpu().numpy()
    pose  = cam.roscamera[1].camera_T_world.detach().cpu().numpy()
    K = cam.roscamera[2].K.detach().cpu().numpy()
    xyz = gaussian.xyz.detach().cpu().numpy()
    # Convert UV coordinates to integer pixel coordinates
    points_homogeneous = np.hstack((xyz, np.ones((xyz.shape[0], 1))))

    points_cam= (pose @ points_homogeneous.T).T[:, :3]
    # points_cam= points_cam_.copy()
    # points_cam[:,0] = points_cam[:,1
    # points_cam[:,1] = - points_cam[:,0]
    # points_cam[:,2] = - points_cam[:,2]


    # uv  = (pose @ points_homogeneous.T).T[:, :3]
    
    # Filter out points outside the image boundaries
    
    # Draw points on the image
    distCoeffs = np.zeros((4, 1))
    points_proj, _ = cv2.projectPoints(np.array(points_cam), np.zeros(3), np.zeros(3),K,distCoeffs)
    points_proj = points_proj.reshape(-1, 2)

    # Draw projected points
    for point in points_proj:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("Gaussian_splatitng check pointcloud", image)
    cv2.waitKey(0)
    cv2.destroyWindow()

def run_visualization_thread(cam,gaussian):
    # Convert uv to NumPy array if it's not already
    # Create a new thread for the visualization
    viz_thread = threading.Thread(target=visualize_uv_on_mat, args=(cam,gaussian))
    
    # Start the thread
    viz_thread.start()

def parameterize(KF):
    # print(f"Input image shape: {KF[0].shape}, dtype: {KF[0].dtype}, range: [{KF[0].min()}, {KF[0].max()}]") 
    
    Cam = RosCamera(image=KF[0], pose=KF[2], downsample_factor=config.downsample_factor, cam_idx=cam_index, device=device, config=config)
    PCL_ros = Pointcloud_ros(depth_image=KF[1], image=KF[0],pose = KF[2] ,device=device, downsample_factor=config.downsample_factor, config=config)
    new_gaussians = PCL_ros.create_gaussians()
    
    # Add debug print for new Gaussians' RGB values
    rgb = new_gaussians.rgb.detach().cpu().numpy()
    white_count = np.sum(np.all(rgb >= 0.99, axis=1))
    total_count = rgb.shape[0]
    print(f"White Gaussian percentage: {white_count/total_count*100:.2f}%")
    print(f"RGB stats - min: {rgb.min()}, max: {rgb.max()}, mean: {rgb.mean()}")
    
    new_gaussians.xyz = torch.nn.Parameter(new_gaussians.xyz)
    new_gaussians.quaternion = torch.nn.Parameter(new_gaussians.quaternion)
    new_gaussians.scale = torch.nn.Parameter(new_gaussians.scale)
    new_gaussians.opacity = torch.nn.Parameter(new_gaussians.opacity)
    new_gaussians.rgb = torch.nn.Parameter(new_gaussians.rgb)
    # run_visualization_thread(Cam,new_gaussians)
    return Cam, new_gaussians

def mapping():
    global trainer
    Cam_KF=[]

    i = 0
    while not cf.mapping_kf:
        time.sleep(1e-15)
    
    newCam, newGaussians = parameterize(cf.mapping_kf[0])


    trainer = SplatTrainer([newCam, newGaussians], config=config)
    Cam_KF.append(newCam)
    trainer.train(i, Cam_KF,True)
    while True:
        time_render = time.time()
        if cf.mapping_kf:
            if cf.new_KF:
                train_idx = cf.new_KF.pop(0)
                newCam, newGaussians = parameterize(cf.mapping_kf[train_idx])
                trainer.densify_postfix(new_gaussians=newGaussians)
                print(f"Added Gaussian: There are {len(newGaussians.xyz)} has been added")
                Cam_KF.append(newCam)
                cf.is_newKF = True
            else:
                cf.is_newKF = False
            trainer.train(i, Cam_KF,cf.is_newKF)
            i += 1   
            # print(f"Count: {len(Cam_KF)}, {len(cf.mapping_kf)}")
            print(f"Render FPS: {1 / (time.time() - time_render)}")
            
            
        else:
            print("case do else *******************************************************************************")    
        
        if len(cf.mapping_kf) > 2000:
            break
    

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
