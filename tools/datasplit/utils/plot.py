import cv2

def plot_BBox(img, tag, boxes):
    """
    :param img: origin image.
    :param tag: result name.
    :param boxes: bounding boxes. Each box is assumed to be in
                  the format [x1FF, y1, x2, y2].
    """
    # 顯示圖片
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow(f'{tag}:{len(boxes)}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

