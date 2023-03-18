import cv2
import random
import argparse
from pathlib import Path

from utils.general import non_max_suppression, get_bounding_boxes
from utils.plot import plot_BBox

def parse_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--Binary_thold', type=int,
                        default=230,
                        help='Binarization threshold')
    parser.add_argument('--nms_thold', type=float,
                        default=0.5,
                        help='non max suppression threshold')
    
    parser.add_argument('--save_result', type=bool,
                        default=False,
                        help='save result')
    
    parser.add_argument('--data_source', type=Path,
                        default=Path('datasets'),
                        help='save result')
    parser.add_argument('--result_path', type=Path,
                        default=Path('result'),
                        help='save result')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    save_result = False
    data_source = args.data_source
    result_path = args.result_path
    (result_path/'labels').mkdir(parents=True,exist_ok=True)
    (result_path/'images').mkdir(parents=True,exist_ok=True)

    for img_path in (data_source/'images').glob('*.bmp'):
        print(img_path.name)
        # 讀入圖片

        img = cv2.imread(str(img_path))
        label = cv2.imread(str(data_source/'labels'/img_path.name))

        for x in range(0,1920,128):
            for y in range(0,1920,128):
                cropped_img = img[y:y+256,x:x+256]
                cropped_label = label[y:y+256,x:x+256]
                cv2.imwrite(str(result_path/'images'/(f'{str(img_path.stem)}_{x//128}_{y//128}.bmp')),cropped_img)
                cv2.imwrite(str(result_path/'labels'/(f'{str(img_path.stem)}_{x//128}_{y//128}.bmp')),cropped_label)

        for i in range(10):
            x = random.randrange(300,1700)
            y = random.randrange(300,1700)
            result_img = img[y:y+256,x:x+256]
            result_label = label[y:y+256,x:x+256]
            rotate = 0
            if random.randrange(2) == 1:
                center = (x+128,y+128)
                rotate = random.randrange(1,72)*5
                M = cv2.getRotationMatrix2D((128,128), rotate, 1.0)
                result_img = cv2.warpAffine(result_img, M, (256, 256))
                result_label = cv2.warpAffine(result_label, M, (256, 256))

            cv2.imwrite(str(result_path/'images'/(f'{str(img_path.stem)}_r{rotate}_{x//128}_{y//128}.bmp')),result_img)
            cv2.imwrite(str(result_path/'labels'/(f'{str(img_path.stem)}_r{rotate}_{x//128}_{y//128}.bmp')),result_label)

        # 取得bbox
        boxes = get_bounding_boxes(label, args.Binary_thold)

        for x1, y1, x2, y2 in boxes:
            for i in range(10):
                size = (x2-x1,y2-y1)
                x = x1 + random.randrange(0,max((100-size[0]),1))
                y = y1 + random.randrange(0,max((100-size[1]),1))
                x = 1792 if x > 1792 else x
                y = 1792 if y > 1792 else y

                result_img = img[y:y+256,x:x+256]
                result_label = label[y:y+256,x:x+256]
                rotate = 0
                if random.randrange(2) == 1:
                    center = (x+128,y+128)
                    rotate = random.randrange(1,72)*5
                    M = cv2.getRotationMatrix2D((128,128), rotate, 1.0)
                    result_img = cv2.warpAffine(result_img, M, (256, 256))
                    result_label = cv2.warpAffine(result_label, M, (256, 256))

                cv2.imwrite(str(result_path/'images'/(f'{str(img_path.stem)}_impurity_{rotate}_{x//128}_{y//128}.bmp')),result_img)
                cv2.imwrite(str(result_path/'labels'/(f'{str(img_path.stem)}_impurity_{rotate}_{x//128}_{y//128}.bmp')),result_label)

            








# if __name__ == '__main__':

#     args = parse_args()
#     save_result = False
#     data_source = args.data_source
#     result_path = args.result_path
#     result_path.mkdir(parents=True,exist_ok=True)

#     for img_path in (data_source/'labels').glob('*.bmp'):
#         print(img_path.name)
#         # 讀入圖片

#         img = cv2.imread(str(img_path))

#         # 取得未nms的bbox
#         boxes = get_bounding_boxes(img, args.Binary_thold)

#         # nms
#         boxes = non_max_suppression(boxes, args.nms_thold)

#         # 顯示圖片
#         result = plot_BBox(img, img_path.stem, boxes)

#         cv2.imwrite(str(result_path/img_path.name),result)