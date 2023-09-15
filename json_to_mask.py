import json
import os
import glob
import shutil
from argparse import ArgumentParser

import numpy as np
import cv2
from PIL import Image

from tqdm import tqdm
import pdb


def parser_args():
    parser = ArgumentParser()
    parser.add_argument('-data_type', type=str)
    return parser.parse_args()


def jsonfile_to_mask(jsonfile_path):
    img = Image.open(jsonfile_path.replace("json", "jpg"))   
    w, h = img.size 
    points = []
    with open(jsonfile_path) as f:
        data = json.loads(f.read())
        for s in data['shapes']:
            points.append(s['points'])
    mask = np.zeros((h, w), dtype=np.uint8())
    for point in points:
        mask = cv2.fillPoly(mask, [np.int32(point).reshape(-1,1,2)], 1)
    return Image.fromarray((mask * 255))


def main():
    opt = parser_args()
    
    bayer_dir = os.path.join(opt.data_type, 'bayer')
    mask_dir = os.path.join(opt.data_type, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    
    json_list = glob.glob(os.path.join(bayer_dir, "*.json"))

    # template
    no = 1
    with open(json_list[no]) as f:
        data = json.loads(f.read())
    shape = data['shapes'][0]
    label = shape['label']
    points = shape['points']
    shape_type = shape['shape_type']
    
    print('[label]', label)
    print('[shape_type]', shape_type)
    print('[points]',points)
    
    for file in tqdm(json_list):
        mask = jsonfile_to_mask(file)
        mask.save(os.path.join(mask_dir, os.path.basename(file).replace("json", "png")))
    print("Mask make finished")

if __name__ == '__main__':
    main()
