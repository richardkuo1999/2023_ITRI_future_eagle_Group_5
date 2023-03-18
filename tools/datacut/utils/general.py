import cv2
import glob
import time
import math
import logging
import numpy as np
from pathlib import Path

import re


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


def get_bounding_boxes(img, Binary_thold):
    """
    :param img: origin image.
    :return: np.array of boxes [[x1,y1,x2,y2],....].
    """
    # 將圖片轉換為灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 將灰度圖進行二值化處理
    _, binary = cv2.threshold(gray, Binary_thold, 255, cv2.THRESH_BINARY)

    # # 顯示原圖和二值化圖
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Binary Image', binary)

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

