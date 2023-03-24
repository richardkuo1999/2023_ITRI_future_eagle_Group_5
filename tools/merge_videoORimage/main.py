import cv2
import numpy as np
from pathlib import Path


imageNumPerLine = 3
result_weigh = 1920
imageType = ['.jpg', '.png', '.bmp']
videoType = ['.mp4']


sourcesPath = Path('data')
resultPath = Path('result')


def addText(image, tag):
    violet = np.zeros((80, image.shape[1], 3), np.uint8)
    violet[:] = (255, 255, 255)
    image = cv2.vconcat((violet, image))
    cv2.putText(image, tag, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, 0)
    return image

def merge_img(tags, expDir, dataName, add_tag):
    image_v = []
    image_h = []
    for j, tag_dirPath in enumerate(zip(tags, expDir), start=1):
        tag, dirPath = tag_dirPath
        SourcePath = dirPath / dataName
        image = cv2.imread(str(SourcePath))
        if str(dirPath).split('/')[-1] =='GT':
            image = addText(image, tag)
        if add_tag:
            image = addText(image, tag)

        image_h.append(image)

        if(j % imageNumPerLine == 0):
            image_v.append(cv2.hconcat(image_h))
            image_h = []
    if(len(image_h) != 0):
        for _ in range(len(image_h),imageNumPerLine):
            image_h.append(np.zeros(image_h[0].shape, dtype=np.uint8))
        image_v.append(cv2.hconcat(image_h))

    image_v = cv2.vconcat(image_v)

    high = int(image_v.shape[0]/(image_v.shape[1]/result_weigh))
    image_v = cv2.resize(image_v, (result_weigh, high),
                            0, 0, cv2.INTER_LINEAR)
    cv2.imwrite(str(resultPath / dataName), image_v)
    
def merge_video(tags, expDir, dataName):
    video_list = []
    for j, tag_dirPath in enumerate(zip(tags, expDir), start=1):
        tag, dirPath = tag_dirPath
        SourcePath = dirPath / dataName

        video = cv2.VideoCapture(str(SourcePath))
        video_list.append(video)

    video_weigh = int(video_list[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    video_hight = int(video_list[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = video_list[0].get(cv2.CAP_PROP_FPS)

    result_height = int(
                int(len(video_list)/imageNumPerLine + \
                (1 if len(video_list)%imageNumPerLine!=0 else 0))* \
                result_weigh/imageNumPerLine/video_weigh * video_hight
                )

    videoWriter = cv2.VideoWriter(str(resultPath/dataName), 
                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, 
                                        (result_weigh, result_height))
    
    successVideo = []
    frameVideo = []
    
    for Video in video_list:
        a, b = Video.read()
        successVideo.append(a)
        frameVideo.append(b)

    while all(successVideo):
        frame_h = []
        frame_v = []
        frames = []
        for f in frameVideo:
            frames.append(cv2.resize(f, (int(result_weigh/imageNumPerLine), 
                    int( result_weigh/imageNumPerLine/video_weigh*\
                        video_hight)), interpolation=cv2.INTER_CUBIC))

        for i, frame in enumerate(frames, start=1):
            frame_h.append(frame)
            if i % imageNumPerLine == 0:
                frame_v.append(np.hstack(tuple(frame_h)))
                frame_h = []

        if(len(frame_h) != 0):
            for _ in range(len(frame_h),imageNumPerLine):
                frame_h.append(np.zeros(frame_h[0].shape, dtype=np.uint8))
            frame_v.append(cv2.hconcat(frame_h))

        frame_result = np.vstack(tuple(frame_v))

        videoWriter.write(frame_result)
        successVideo = []
        frameVideo = []
        for Video in video_list:
            a, b = Video.read()
            successVideo.append(a)
            frameVideo.append(b)
    videoWriter.release()
    for Video in video_list:
        Video.release()
    video_list = []
        
if __name__ == '__main__':
    dataType = imageType + videoType
    resultPath.mkdir(exist_ok=True)
    add_tag = False

    expDir = [dir for dir in sourcesPath.iterdir()]
    tags = [str(dir).split('\\')[-1].split('(')[0] for dir in expDir]
    expNum = len(tags)
    expData = (
        data.name for data in expDir[0].iterdir() if data.suffix in dataType)

    for i, dataName in enumerate(expData):
        print(i, dataName)

        if(Path(dataName).suffix in videoType):
            merge_video(tags, expDir, dataName)
        elif(Path(dataName).suffix in imageType):
            merge_img(tags, expDir, dataName, add_tag)
