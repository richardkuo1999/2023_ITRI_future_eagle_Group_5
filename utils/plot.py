import cv2
import numpy as np

def plot_BBox(img, tag, boxes, save=False):
    """
    :param img: origin image.
    :param tag: result name.
    :param boxes: bounding boxes. Each box is assumed to be in
                  the format [x1FF, y1, x2, y2].
    """
    # 顯示圖片
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    if save:
        cv2.imwrite(f'{tag}_{len(boxes)}.bmp', img)
    cv2.imshow(f'{tag}:{len(boxes)}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_seg_result(img, result, index, epoch, save_dir=None,palette=None,is_demo=False,is_gt=False):
    
    color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[result == label, :] = color



    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    # FIXME　draw result
    img[color_mask == 0] = img[color_mask == 0] * 0.3
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + [0,125,0]
    # img[color_mask != 0] = color_seg[color_mask != 0] * 0.5
    # img[color_mask == 0] = img[color_mask == 0] * 0.2
    # img[color_mask != 0] = color_seg[color_mask != 0] * 0.8

    img = img.astype(np.uint8)
    img = cv2.resize(img, (2048,2048), interpolation=cv2.INTER_LINEAR)

    if not is_demo:
        if not is_gt:
            cv2.imwrite(save_dir+"/batch_{}_{}_segresult.png".format(epoch,index), img)
        else:
            cv2.imwrite(save_dir+"/batch_{}_{}_seg_gt.png".format(epoch,index), img)
    return img