from re import L
import numpy
import os
import sys
root = '/home/liushiqi/liusq/AB3DMOT/data/KITTI/tracking/training/calib'
file_names = os.listdir(root)
file_ob_list = []
for file_name in file_names:
    fileob = root + '/' + file_name
    file_ob_list.append(fileob)
for file1 in file_ob_list:
    print(file1)
    with open(file1,"r+") as f:
        file = f.read()
        file = file.replace('R_rect','R0_rect')
        file = file.replace('Tr_velo_cam','Tr_velo_to_cam')
        file = file.replace('Tr_imu_velo','Tr_imu_to_velo')
        print(file)
        f.seek(0,0)
        f.truncate()	#清空文件，配合seek使用，否则清空的位置不对
        f.write(file)
        f.close()