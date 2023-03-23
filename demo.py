import cv2
import yaml
import time
import logging
import argparse
import numpy as np
from numpy import random
from pathlib import Path

import torch
import torchvision.transforms as transforms


# from models.YOLOP import Model
from utils.datasets import LoadImages
from utils.plot import show_seg_result
from utils.torch_utils import select_device, time_synchronized
from utils.general import colorstr, increment_path, write_log,\
                         data_color, AverageMeter, OpCounter, addText2image
from models.model import build_model


logger = logging.getLogger(__name__)

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(args, device, expName):

    save_dir = args.save_dir

    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    results_file = save_dir / 'results.txt'
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Get class and class number
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    class_name = data_dict['class_name']
    num_classes = len(class_name)
    logger.info(f"{colorstr('class: ')}{class_name}")
    class_color = data_color(class_name)

    # Load model
    model = build_model(args.cfg, num_classes).to(device)
    checkpoint = torch.load(args.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # calculate macs, params, flops, parameter count
    img = np.random.rand(384, 640, 3)
    img = transform(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    OpCounter(img, model, results_file)
    
    # Set Dataloader
    dataset = LoadImages(args.source, img_size=args.img_size)
    bs = 1  # batch_size

    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None

    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    for i, (path, img, img_det, vid_cap,shapes) in enumerate(dataset):
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        output = model(img)
        t2 = time_synchronized()
        inf_time.update(t2-t1,img.size(0))


        save_path = save_dir / Path(path).name \
            if dataset.mode != 'stream' else save_dir / "web.mp4"

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = output[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, 
                                        scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

      
        img_det = show_seg_result(img_det, da_seg_mask, _, _, 
                            palette=class_color, is_demo=True)

        fps= round(1/(inf_time.val+nms_time.val))
        print(f'FPS:{fps}')
        img_det = addText2image(img_det, expName,fps)
        if dataset.mode == 'image':
            cv2.imwrite(str(save_path),img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(str(save_path), 
                                    cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)

    msg = f'{str(args.weights)}\n'+\
          f'Results saved to {str(args.save_dir)}\n'+\
          f'Done. ({(time.time() - t0)} s)\n'+\
          f'inf : ({inf_time.avg} s/frame)\n'+\
          f'fps : ({(1/(inf_time.avg+nms_time.avg))} frame/s)'
    print(msg)
    write_log(results_file, msg)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logDir', type=str, default='runs/demo',
                            help='log directory')
    parser.add_argument('--cfg', type=str, default='', 
                                            help='model yaml path')
    parser.add_argument('--weights', type=str, default='', 
                                                    help='model.pth path(s)')
    parser.add_argument('--data', type=str, default='data/full.yaml', 
                                            help='dataset yaml path')
    parser.add_argument('--source', type=str, default='./inference/images', 
                                                    help='source')  
    parser.add_argument('--img-size', type=int, default=2048, 
                                                    help='inference size (pixels)')
    parser.add_argument('--device', default='0', 
                                    help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()

    device = select_device(args.device)
    

    if args.weights != '':
        args.save_dir = increment_path(Path(args.logDir))  # increment run
        tag = args.cfg if args.cfg.split('.')[-1] != 'yaml' else args.cfg.split('.')[0]

        with torch.no_grad():
            detect(args, device, tag)
    else:
        for cfg in Path('weights').glob('*/*.pth'):

            args.cfg = cfg.parts[1]
            args.weights = cfg
            if args.cfg in ['YOLOv7_b3','YOLOv7_bT2']:
                args.cfg += '.yaml'

            tag = args.cfg if args.cfg.split('.')[-1] != 'yaml' else args.cfg.split('.')[0]
            args.save_dir = increment_path(Path(args.logDir)/tag)  # increment run

            with torch.no_grad():
                detect(args, device, tag)