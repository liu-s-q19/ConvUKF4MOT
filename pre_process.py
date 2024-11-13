from re import L
import numpy
import os
import sys
# 输入文件夹
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
        file = file.replace('R0_rect::','R0_rect:')
        file = file.replace('Tr_velo_to_cam::','Tr_velo_to_cam:')
        file = file.replace('Tr_imu_to_velo::','Tr_imu_to_velo:')
        print(file)
        f.seek(0,0)
        f.truncate()	#清空文件，配合seek使用，否则清空的位置不对
        f.write(file)
        f.close()
        # for line in f:
        #     xiugai = line.split(' ')
        #     if(xiugai[0]=='R_rect'):
        #         line = line[0:6]+':'+line[6:]
        #     if(xiugai[0]=='Tr_velo_cam'):
        #         line = line[0:11]+':'+line[11:]
        #     if(xiugai[0]=='Tr_imu_velo'):
        #         line = line[0:11]+':'+line[11:]    