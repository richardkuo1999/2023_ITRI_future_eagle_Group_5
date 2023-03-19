import cv2
import copy
import json
import shutil
import argparse
from pathlib import Path
from matplotlib import pyplot as plt

from utils.general import split_data, get_bounding_boxes, write_bbox_to_txt
from utils.plot import plot_BBox

coco_format = {
              "categories":[],
              "annotations": [],
              "images": []
              }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=Path,
                        default=Path('datasets'))
    parser.add_argument('--save_path', type=Path,
                        default=Path('result'))
    
    parser.add_argument('--get_yolo_type', type=bool,
                        default=True)
    parser.add_argument('--get_coco_type', type=bool,
                        default=True)
    parser.add_argument('--debug', type=bool,
                        default=False,
                        help='show image')
    
    parser.add_argument('--class_names', type=list,
            default=['metal', 'NGfish', 'rope', 'seafood', 'stone'])
    parser.add_argument('--proportion', type=list,
                        default=[6,1],
                        help='train data : val data')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    source_path, save_path = args.source_path, args.save_path
    class_names, proportion = args.class_names, args.proportion
    train_id, val_id = -1, -1

    if args.get_coco_type:
        train_coco = copy.deepcopy(coco_format)
        for i, name in enumerate(class_names):
           categories = {"id": i, "name": name, "supercategory": "mark"}
           train_coco['categories'].append(categories)
        val_coco = copy.deepcopy(train_coco)

    for target in ['train', 'val']:
      (save_path / 'images' / target).mkdir(exist_ok= True,parents=True)
      (save_path / 'labels' / target).mkdir(exist_ok= True,parents=True)

    flag = False if input("Folders by Category[y/n]:") == 'n' else True

    with open(str(save_path / 'labels' / 'train' / 'classes.txt'),"a") as f:
        for name in class_names:
          f.write(f'{name}\n')
    shutil.copy(str(save_path / 'labels' / 'train' / 'classes.txt'),
                  str(save_path / 'labels' / 'val' / 'classes.txt'))

    for dataclass in source_path.iterdir():
        class_id = None
        if dataclass.name in class_names:
          class_id = class_names.index(dataclass.name)
        if flag:
          for target in ['train', 'val']:
            (save_path / 'images' / target / dataclass.name).mkdir(exist_ok= True,parents=True)
            (save_path / 'labels' / target / dataclass.name).mkdir(exist_ok= True,parents=True)
          
        
        # rename the label file
        for original in (dataclass/'label').glob('*.bmp'):
          new = original.parent / ((original.stem).split('_')[0]+'.bmp')
          shutil.move(original , new) # image

        # split data
        train_image_list = split_data(dataclass, proportion)

        # copy file
        for data in dataclass.glob('*.bmp'):
            move_to = Path('train' if data.stem in train_image_list else 'val')
            path_type = move_to / dataclass.name if flag else move_to

            shutil.copy(data, save_path /'images' / path_type /data.name)
            label_path =  dataclass / 'label' / data.name
            shutil.copy(label_path, save_path / 'labels' / path_type / data.name)

            if args.get_yolo_typ or args.get_coco_type:
                img = cv2.imread(str(label_path))
               
            '''get_yolo_type'''
            if args.get_yolo_type:
                # # get bbox
                boxes = get_bounding_boxes(img)
                if args.debug:
                  plot_BBox(img.copy(), f'{data.stem} origin box', boxes)

                # write to .txt
                txt_path = save_path / 'labels'  / path_type / (str(data.stem)+'.txt')
                write_bbox_to_txt(img, str(txt_path), boxes, class_id)


            '''get_coco_type'''
            if args.get_coco_type:
                h, w = img.shape[:2]
                coco_append = None
                if str(move_to) == 'train':
                  train_id+=1
                  image_id = train_id
                  coco_append = train_coco
                else:  
                  val_id+=1
                  image_id = val_id
                  coco_append = val_coco
                images = {"file_name": data.name, "id": image_id, "width": w, "height": h}
                coco_append['images'].append(images)
    if args.get_coco_type:
      coco_output = save_path / 'labels' / 'train' / 'output.json'
      with open(coco_output, 'w') as f:
          json.dump(train_coco, f)
      coco_output = save_path / 'labels' / 'val' / 'output.json'
      with open(coco_output, 'w') as f:
          json.dump(val_coco, f)
          
          
                
                

