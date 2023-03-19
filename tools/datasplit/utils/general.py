import cv2
import random
import numpy as np


def split_data(dataclass, Proportion):
    image_list = [p.stem for p in (dataclass).glob('*.bmp')]
    k = len(image_list) // sum(Proportion) * Proportion[0]
    print(f'{dataclass.name}:{k}')

    return random.sample(image_list, k=k)

def get_bounding_boxes(img):
    """
    :param img: origin image.
    :return: np.array of boxes [[x1,y1,x2,y2],....].
    """
    # 將圖片轉換為灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 進行輪廓檢測
    contours, hierarchy = cv2.findContours(
        gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 框框
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x+w, y+h])
    return np.array(boxes)


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def write_bbox_to_txt(img, output_path, boxes, classes):
    H, W = img.shape[:2]
    with open(output_path,"w") as f:
        for box in boxes:
            x, y, w, h = convert((W,H), box)
            f.write(f'{classes} {x} {y} {w} {h}\n')

