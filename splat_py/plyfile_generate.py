
import numpy as np
from plyfile import PlyElement, PlyData # Requires plyfile==0.8.1
import torch
import csv
def construct_list_of_attributes(_features_dc, _scaling, _rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(_features_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    # for i in range(_features_rest.shape[1]):
    #     l.append('f_rest{}'.format(i))
    l.append('opacity')
    for i in range(_scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(_rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


def save_ply(path, gaussian):
    # makedirs(os.path.dirname(path), exist_ok= True)
    xyz = gaussian.xyz.detach().cpu().numpy()
    xyz = np.column_stack((-xyz[:, 2], xyz[:, 1], xyz[:, 0]))

    
    normals = np.zeros_like(xyz)
    rgb = gaussian.rgb.detach().cpu().numpy()  * 0.28209479177387814 #+0.5
    rgb = rgb.astype(np.uint8)
    opacities = torch.sigmoid(gaussian.opacity).detach().cpu().numpy()
    print(f"RGB range: min={rgb.min()}, max={rgb.max()}")
    print(f"Opacity range: min={opacities.min()}, max={opacities.max()}")
    scale = gaussian.scale.detach().cpu().numpy()
    rotation = gaussian.quaternion.detach().cpu().numpy()
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(rgb, scale, rotation)]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, rgb, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    # PlyData([el],text = True).write(path)
    PlyData([el]).write(path)

def writeRGB(rgb):
    with open('rgbDEBUG.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Optional: Write the header
        # writer.writerow(['col1', 'col2', 'col3'])
        # Write data rows
        writer.writerows(rgb)

    print("Data written to output.csv")

def writeOPACITY(opacity):
    with open('OPACITY.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Optional: Write the header
        # writer.writerow(['col1', 'col2', 'col3'])
        # Write data rows
        writer.writerows(opacity)

    print("Data written to output.csv")