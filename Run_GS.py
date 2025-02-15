import rospy
import os
import cv2
import torch
import time
import torch.multiprocessing as mp
import torch.multiprocessing
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
from splat_py.sub_data import Subdata
from splat_py.optimizer_manager import OptimizerManager
from splat_py.structs import GSMetrics
from splat_py.rasterize import rasterize
from splat_py.utils import (
    inverse_sigmoid,
    quaternion_to_rotation_torch,
)
import config as cf
import tyro
import yaml

device= torch.device("cuda:0")
config = tyro.cli(SplatConfigs)
if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)
# save a copy of the config
yaml.dump(config, open(os.path.join(config.output_dir, "config.yaml"), "w"))


# Variable
cam_index=0
stack_gaussian=None
cf.mapping_kf=[]
cf.new_gaussians =None 
cf.is_newKF=False
cf.using_sync=True
lock = Lock()
trainer=None
cf.depth_image =None
cf.img =None 
cf.odometry=None
cf.new_KF=[]
def _callback(depth_image,image,odometry):
    global cam_index,trainer
    cf.depth_image =depth_image
    cf.img =image 
    cf.odometry=odometry
   
    if cf.depth_image is not None and cf.img is not None and cf.odometry is not None:
        if cam_index == 0:
            cf.mapping_kf.append([image,depth_image,odometry])
            cf.new_KF.append(len(cf.mapping_kf)-1)
        if cam_index%10 == 0 and cam_index > 0:
            cf.mapping_kf.append([image,depth_image,odometry])
            cf.is_newKF=True
            cf.new_KF.append(len(cf.mapping_kf)-1)
            print("new KF here")
        cam_index+=1
    print(f"Success ! {cam_index}")


def parameterize(KF):
    # if cf.depth_image is not None and cf.img is not None and cf.odometry is not None:
        Cam = RosCamera(image=KF[0],pose=KF[2],downsample_factor=config.downsample_factor,\
                cam_idx=cam_index, device=device, config=config)
        # pcl_as_numpy_array = np.array(list(pc2.read_points(pcl, field_names=("x", "y", "z")))
        PCL_ros=Pointcloud_ros(depth_image=KF[1],
                            image=KF[0],
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
        return Cam,new_gaussians
def mapping():
    global train_mode
    train_mode=False 
    i=0
    trainer = None
    while len(cf.mapping_kf)==0:
        time.sleep(1e-15)
    newCam,newGaussians=parameterize(cf.mapping_kf[0])
    trainer = SplatTrainer([newCam,newGaussians],config= config)
    trainer.train(i,newCam)
                
    while True:  
            
            #**** return a list with  Cam.roscamera[1] posed image, Cam.roscamera[2] image param/matrix, cam_idx=Cam.roscamera[0]
            # if len(cf.mapping_kf) != 0 and train_mode==False:
            #     print("initialize********************************************************")
            #     # newCam,newGaussians= cf.mapping_kf[0]
            #     newCam,newGaussians=parameterize(cf.mapping_kf[0])
            #     trainer = SplatTrainer([newCam,newGaussians],config= config)
                
            #     trainer.train(i,newCam)
                
            #     train_mode = True
            #     # break
            # if cf.is_newKF:
            #     trainer.densify_postfix(new_gaussians=newGaussians)
            #     print("add Gaussian")
            #     cf.is_newKF=False
            
            if len(cf.mapping_kf)>0:
                if len(cf.new_KF)>0:
                    train_idx = cf.new_KF.pop(0)
                    newCam,newGaussians=parameterize(cf.mapping_kf[train_idx])
                    trainer.densify_postfix(new_gaussians=newGaussians)
                    print("add Gaussian")
                    # trainer.train(i,newCam)
                else:
                    train_idx = random.choice(range(len(cf.mapping_kf)))
                    newCam,newGaussians= parameterize(cf.mapping_kf[train_idx]) 
                print("count", len(cf.mapping_kf))
                time_render=time.time()
                trainer.train(i,newCam)
                print("render_time _10",1/(time.time()-time_render) )
            # if train_mode == True:
            # #     # newCam,newGaussians= cf.mapping_kf[0]
            # #     # trainer.train(i,newCam)
            #     # len_of_KF=cf.mapping_kf
            #     if cf.is_newKF:

            #         newCam,newGaussians=parameterize(cf.mapping_kf[-1])
            #         # newCam,newGaussians= cf.mapping_kf[-1]
            #         trainer.densify_postfix(new_gaussians=newGaussians) #, clone_mask=np.int64(len(newGaussians.xyz))
            #         trainer.train(i,newCam)
                   
            #         cf.is_newKF=False
            #     else: 
            #         train_idx = random.choice(range(len(cf.mapping_kf)))
            #         newCam,newGaussians= parameterize(cf.mapping_kf[train_idx]) 
            #         trainer.train(i,newCam)
                
            i+=1
            if len(cf.mapping_kf)>100:
                 break
            # rospy.sleep(rospy.Duration(nsecs=1000))

def listener():
    rospy.init_node('GS_Map')


    if cf.using_sync:
        subscribers = [
            # message_filters.Subscriber("/velodyne_points",PointCloud2),
            image_transport.Subscriber("/zed2/zed_node/depth/depth_registered"),
            # message_filters.Subscriber("/lvi_sam/vins/depth/depth_cloud",PointCloud2),
            image_transport.Subscriber("/zed2/zed_node/left_raw/image_raw_color"),
            message_filters.Subscriber("/lvi_sam/lidar/mapping/odometry",Odometry)
        ]

        mf = message_filters.ApproximateTimeSynchronizer(
            subscribers,
            queue_size=10,
            slop=0.1)#0.03

        mf.registerCallback(_callback)
        
    # else:
    #     sub=Subdata()
        # if sub.PCL_ros is not None:
            
        #     if cam_index == 0:
        #         cf.mapping_kf.append([sub.Cam,new_gaussians])
        # #     # trainer = SplatTrainer(mapping_kf,config= config)
        #         trainer = SplatTrainer(cf.mapping_kf[0],config= config)
        #         newCam,newGaussians= cf.mapping_kf[0]
        #         trainer.train(cam_index,newCam)
        #     if cam_index%5 == 0 and cam_index!=0:
        #         cf.mapping_kf.append([sub.Cam,new_gaussians])
        #         cf.is_newKF=True
                
        #         newCam,newGaussians= cf.mapping_kf[-1]
        #         # trainer = SplatTrainer(cf.mapping_kf[-1],config= config)
        #         trainer.densify_postfix(new_gaussians=newGaussians, clone_mask=np.int64(len(newGaussians.xyz)))
        #         print(f"NO of gaussian map {len(trainer.gaussians.xyz)}")
        #         t=time.time()
        #         trainer.train(cam_index,newCam)
        #         print("FPS test", 1/(time.time()-t))

        #     print(f"Success ! {cam_index}")
        #     cam_index+=1
    # rospy.spin()
        
if __name__ == '__main__':
    
    mapping=Thread(target=mapping)
    mapping.start()
    # p = mp.Process(target=mapping) 
    # p.start()
    # p.join()
  
    listener()
    
    
    # run()
    # mapping()
   
    # 
    
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print("Shutting down")
    # trainer = SplatTrainer(mapping_kf,config= config)
    # 
