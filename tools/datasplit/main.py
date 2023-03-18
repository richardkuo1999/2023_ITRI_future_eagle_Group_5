import cv2
import shutil
import argparse
from pathlib import Path
from matplotlib import pyplot as plt

from utils.general import split_data, get_bounding_boxes, write_bbox_to_txt





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
    
    parser.add_argument('--class_name', type=list,
            default=['metal', 'NGfish', 'rope', 'seafood', 'stone'])
    parser.add_argument('--proportion', type=list,
                        default=[6,1],
                        help='train data : val data')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    source_path, save_path = args.source_path, args.save_path
    class_name, proportion = args.class_name, args.proportion

    (save_path / 'images' / 'train').mkdir(exist_ok= True,parents=True)
    (save_path / 'labels' / 'train').mkdir(exist_ok= True,parents=True)
    (save_path / 'images' / 'val').mkdir(exist_ok= True,parents=True)
    (save_path / 'labels' / 'val').mkdir(exist_ok= True,parents=True)

    flag = False if input("Folders by Category[y/n]:") == 'n' else True

    with open(str(save_path / 'labels' / 'train' / 'classes.txt'),"a") as f:
        for name in class_name:
          f.write(f'{name}\n')
    shutil.copy(str(save_path / 'labels' / 'train' / 'classes.txt'),
                  str(save_path / 'labels' / 'val' / 'classes.txt'))

    for dataclass in source_path.iterdir():
        class_id = None
        if dataclass.name in class_name:
          class_id = class_name.index(dataclass.name)
        if flag:
          (save_path / 'images' / 'train' / dataclass.name).mkdir(exist_ok= True,parents=True)
          (save_path / 'labels' / 'train' / dataclass.name).mkdir(exist_ok= True,parents=True)
          (save_path / 'images' / 'val' / dataclass.name).mkdir(exist_ok= True,parents=True)
          (save_path / 'labels' / 'val' / dataclass.name).mkdir(exist_ok= True,parents=True)
          


        
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


            '''get_yolo_type'''
            if args.get_yolo_type:
                img = cv2.imread(str(label_path))
                # # get bbox
                bbox = get_bounding_boxes(img)
                # write to .txt
                txt_path = save_path / 'labels'  / path_type / (str(data.name)+'.txt')
                write_bbox_to_txt(img, str(txt_path), bbox, class_id, False)

            # '''get_coco_type'''
            # if args.get_coco_type:



