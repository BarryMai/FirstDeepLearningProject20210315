# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Created on Thu may 18 15:46:16 2021
@author: mxc
"""

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import sys
import  imgviz
import numpy as np
import PIL.Image
import labelme
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', help='input annotated directory',
        type=str,
        default='inputJson')
    parser.add_argument('--output_dir', help='output dataset directory',
        type=str,
        default='output')
    parser.add_argument('--labels', help='labels file', required=False,
        type=str,
        default='labels.txt')
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
    #        sys.exit(1)
    else:
        os.makedirs(args.output_dir)
        os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
        os.makedirs(osp.join(args.output_dir, 'SegmentationClass'))
        #    os.makedirs(osp.join(args.output_dir, 'SegmentationClassPNG'))
        os.makedirs(osp.join(args.output_dir, 'SegmentationClassVisualization'))
    saved_path = args.output_dir
    if not os.path.exists(os.path.join(saved_path, 'ImageSets', 'Segmentation')):
        os.makedirs(os.path.join(saved_path, 'ImageSets', 'Segmentation'))
    print('Creating dataset:', args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        print(i)
        class_id = i + 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(args.output_dir, 'class_names.txt')

    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    #colormap = labelme.utils.label_colormap(255)


    for label_file in glob.glob(os.path.join(args.input_dir, '*.json')):
        print('Generating dataset from:', label_file)
        try:
            with open(label_file) as f:
                base = os.path.splitext(osp.basename(label_file))[0]
                out_img_file = os.path.join(
                    args.output_dir, 'JPEGImages', base + '.jpg')
                print(out_img_file)
                #            out_lbl_file = osp.join(
                #                args.output_dir, 'SegmentationClass', base + '.npy')
                #                args.output_dir, 'SegmentationClass', base + '.npy')
                out_png_file = os.path.join(
                    args.output_dir, 'SegmentationClass', base + '.png')
                out_viz_file = os.path.join(
                    args.output_dir,
                    'SegmentationClassVisualization',
                    base + '.jpg',
                )

                data = json.load(f)

                img_file = os.path.join(label_file.split('.json')[0] + '.bmp')
                imageData = data.get("imageData")
                #img = np.asarray(PIL.Image.open(img_file))
                img = labelme.utils.img_b64_to_arr(imageData)

                PIL.Image.fromarray(img).save(out_img_file)

                print('class_name_to_id:', class_name_to_id)
                # lbl = labelme.utils.shapes_to_label(
                #     img_shape=img.shape,
                #     shapes=data['shapes'],
                #     label_name_to_value=class_name_to_id,
                # )
                for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                    label_name = shape["label"]
                    if label_name in class_name_to_id:
                        label_value = class_name_to_id[label_name]
                    else:
                        label_value = len(class_name_to_id)
                        class_name_to_id[label_name] = label_value
                lbl, _ = labelme.utils.shapes_to_label(
                    img.shape, data["shapes"], class_name_to_id
                )

                labelme.utils.lblsave(out_png_file, lbl)
                #            np.save(out_lbl_file, lbl)

                # viz = labelme.utils.draw_label(
                #     lbl, img, class_names, colormap=colormap)
                label_names = [None] * (max(class_name_to_id.values()) + 1)
                viz = imgviz.label2rgb(
                    label=lbl, img=imgviz.asgray(img), label_names=label_names, loc="rb"
                )
                PIL.Image.fromarray(viz).save(out_viz_file)
        except:
            with open('wrongdata.txt', 'w') as f:
                f.write(label_file + '\n')
            print('这张图像有错误')
            continue

    # 6.split files for txt
    txtsavepath = os.path.join(saved_path, 'ImageSets', 'Segmentation')
    ftrainval = open(os.path.join(txtsavepath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')
    ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
    fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')
    total_files = os.listdir(os.path.join(args.output_dir, 'SegmentationClass'))
    total_files = [i.split("/")[-1].split(".png")[0] for i in total_files]
    # test_filepath = ""
    for file in total_files:
        ftrainval.write(file + "\n")
    # test
    # for file in os.listdir(test_filepath):
    #    ftest.write(file.split(".jpg")[0] + "\n")
    # split
    train_files, val_files = train_test_split(total_files, test_size=0.5, random_state=42)
    # train
    for file in train_files:
        ftrain.write(file + "\n")
    # val
    for file in val_files:
        fval.write(file + "\n")

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


if __name__ == '__main__':
    main()