import os

path = 'D:\GitDownLoad\PaddleSeg\data\optic_disc_seg\mai\pic'  # path为json文件存放的路径
json_file = os.listdir(path)
for file in json_file:
    os.system("D:\programFiles\anaconda\envs\labelme\Scripts/labelme_json_to_dataset.exe %s"%(path + '/' + file))