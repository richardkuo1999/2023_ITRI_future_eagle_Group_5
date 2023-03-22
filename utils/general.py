import cv2
import glob
import time
import math
import logging
import numpy as np
from pathlib import Path

import re

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        
def set_logging():
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO)
    
def write_log(results_file, msg):
    with open(results_file, 'a') as f:
        f.write(msg+'\n')  

def value_to_float(x):
    if type(x) == float or type(x) == int or not x:
        return x
    if 'K' == x[-1]:
        if len(x) > 1:
            return float(x.replace('K', '')) * 1000
        return 1000.0
    if 'M' == x[-1]:
        if len(x) > 1:
            return float(x.replace('M', '')) * 1000000
        return 1000000.0
    if 'G' == x[-1]:
        return float(x.replace('G', '')) * 1000000000
    return x

def OpCounter(img, model, results_file):
    """get macs, params, flops, parameter count

    Args:
        img (torch.Tensor): Test data
        model (models): Test model
        results_file (pathlib): save resuslt
    """
    macs, params = profile(model, inputs=(img, ))  # ,verbose=False

    write_log(results_file, f"MACs: {macs*2}")
    write_log(results_file, f"params: {params}")


    flops = FlopCountAnalysis(model, img)
    write_log(results_file, f"FLOPs: {flops.total()}")

    # write results to csv
    def write_csv(fileName, table):
        parameter_data = table.split('\n')

        data = {}
        for i, index in enumerate(parameter_data[0].split('|')[1:-1], start=1):
            data[index.strip(' ')] = [value_to_float(line.split('|')[i].strip(' ')) for line in parameter_data[2:]]

        myvar = pd.DataFrame(data)
        myvar.to_csv(str(results_file).replace("results.txt",fileName))


    parameter_table = parameter_count_table(model)
    write_csv("parameter.csv", parameter_table)
    
    flop_table = flop_count_table(flops)
    write_csv("flop.csv", flop_table)

def addText2image(image, tag, fps):
                """add Tag and fps to image

                Args:
                    image (_type_): image
                    tag (str): model name
                    fps (int): fps

                Returns:
                    _type_: image
                """
                violet = np.zeros((80, image.shape[1], 3), np.uint8)
                violet[:] = (255, 255, 255)
                image = cv2.vconcat((violet, image))
                cv2.putText(image, f'{tag},  FPS:{fps}', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, 0)
                return image


def one_hot_it_v11_dice(label, label_info):
    # return semantic_map -> [H, W, class_num]
    semantic_map = []
    # void = np.zeros(label.shape[:2])
    for index, info in enumerate(label_info):
        color = label_info[info][:3]
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        # semantic_map[class_map] = index
        semantic_map.append(class_map)
        
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map

def non_max_suppression(boxes, threshold):
    """
    :param boxes: bounding boxes to perform NMS on. Each box is assumed to be in
                  the format [x1, y1, x2, y2].
    :param threshold: intersection-over-union (IoU) threshold for overlapping
                      boxes.
    :return: list of indices of boxes that were kept after NMS.
    """
    if len(boxes) == 0:
        return []

    # Convert boxes to numpy array
    boxes = np.array(boxes)

    # Compute box areas
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    # Sort boxes by bottom-right y-coordinate
    y2 = boxes[:, 3]
    idxs = y2.argsort()

    # Initialize the list of picked indices
    picked_idxs = []

    # Keep looping while some indices still remain in the indices list
    while len(idxs) > 0:
        # Grab the last index in the indices list and add it to the picked list
        last = len(idxs) - 1
        i = idxs[last]
        picked_idxs.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box
        # and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[:last], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[:last], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[:last], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[:last], 3])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the overlap between the bounding boxes
        overlap = (w * h) / areas[idxs[:last]]

        # Remove indices of overlapping boxes
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > threshold)[0])))

    return boxes[picked_idxs]


def get_bounding_boxes(img, Binary_thold, save=False):
    """
    :param img: origin image.
    :return: np.array of boxes [[x1,y1,x2,y2],....].
    """
    # 將圖片轉換為灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 將灰度圖進行二值化處理
    _, binary = cv2.threshold(gray, Binary_thold, 255, cv2.THRESH_BINARY)

    # # 顯示原圖和二值化圖
    if save:
        cv2.imwrite('Original Image.bmp', img)
        cv2.imwrite('Binary Image.bmp', binary)
    cv2.imshow('Original Image', img)
    cv2.imshow('Binary Image', binary)

    # 進行輪廓檢測
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 框框
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # TODO not the best way
        if((x > 0 or y > 0) and (w < 500 and h < 500)):
            boxes.append([x, y, x+w, y+h])
    return np.array(boxes)

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    path = path / time_str
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return path
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return path / sep / n  # update path
    
def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *opt, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in opt) + f'{string}' + colors['end']

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def data_color(label_info):
    color = []
    for index, info in enumerate(label_info):
        color.append(np.array(label_info[info][:3]))
    return color