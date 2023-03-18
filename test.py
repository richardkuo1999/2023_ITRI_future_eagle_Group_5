import cv2
import yaml
import json
import random
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch



from utils.loss import Loss
from utils.datasets import create_dataloader
from utils.torch_utils import select_device, time_synchronized
from utils.plot import show_seg_result
from utils.metrics import SegmentationMetric
from utils.general import colorstr, increment_path, write_log,\
                        check_img_size, data_color, AverageMeter
from models.model import build_model

logger = logging.getLogger(__name__)

def test(epoch, args, hyp, val_loader, model, criterion, output_dir,
              results_file, class_name, class_color, logger=None, 
                                                        device='cpu'):

    # setting
    max_stride = 32
    weights = None

    save_dir = output_dir / 'visualization'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir = str(save_dir)

     #imgsz is multiple of max_stride
    _, imgsz = [check_img_size(x, s=max_stride) for x in args.img_size]
    batch_size = args.batch_size
    save_hybrid=False

    #iou vector for mAP@0.5:0.95
    iouv = torch.linspace(0.5,0.95,10).to(device)    
    niou = iouv.numel()

    num_classes = hyp['num_classes']
    seen =  0 
    metric = SegmentationMetric(num_classes) #drive area segment confusion matrix    

    class_name = list(class_name.keys())

    
    t_inf = 0.
    
    losses = AverageMeter()

    acc_seg = AverageMeter()
    IoU_seg = AverageMeter()
    mIoU_seg = AverageMeter()

    T_inf = AverageMeter()

    # switch to train mode
    model.eval()

    for batch_i, (img, target, paths, shapes) in enumerate(tqdm(val_loader)):

        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        nb, _, height, width = img.shape    #batch size, channel, height, width

        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = min(shapes[0][1][0])

            t = time_synchronized()
            outputs = model(img)
            t_inf = time_synchronized() - t
            if batch_i > 0:
                T_inf.update(t_inf/img.size(0),img.size(0))

            loss = criterion(outputs, target)   #Compute loss
            losses.update(loss.item(), img.size(0))


            if batch_i == 0:
                for i in range(batch_size):
                    img_test = cv2.imread(paths[i])
                    seg_mask = outputs[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                    seg_mask = torch.nn.functional.interpolate(seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                    _, seg_mask = torch.max(seg_mask, 1)

                    gt_mask = target[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                    gt_mask = torch.nn.functional.interpolate(gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                    _, gt_mask = torch.max(gt_mask, 1)

                    seg_mask = seg_mask.int().squeeze().cpu().numpy()
                    gt_mask = gt_mask.int().squeeze().cpu().numpy()
                    # seg_mask = seg_mask > 0.5
                    # plot_img_and_mask(img_test, seg_mask, i,epoch,save_dir)
                    img_test1 = img_test.copy()
                    _ = show_seg_result(img_test, seg_mask, i,epoch, save_dir, palette=class_color)
                    _ = show_seg_result(img_test1, gt_mask, i, epoch, save_dir, palette=class_color
                                                                                            , is_gt=True)



        #driving area segment evaluation
        _,predict=torch.max(outputs, 1)
        _,gt=torch.max(target, 1)
        predict = predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
        gt = gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

        metric.reset()
        metric.addBatch(predict.cpu(), gt.cpu())
        acc = metric.pixelAccuracy()
        IoU, mIoU = metric.IntersectionOverUnion()

        acc_seg.update(acc,img.size(0))
        IoU_seg.update(IoU,img.size(0))
        mIoU_seg.update(mIoU,img.size(0))



    model.float()  # for training


    segment_result = (acc_seg.avg,IoU_seg.avg,mIoU_seg.avg)


    t = T_inf.avg

    # Print results
    msg = f'Epoch: [{epoch}]    Loss({losses.avg:.3f})\nDetect:\n'

    if  num_classes > 1:
        pf = '%20s' + '%13g' # print format
        msg += 'Driving area Segment:\n'
        msg += (('%20s' + '%13s') % ('class', 'IoU')+'\n')
        for i, iou in enumerate(IoU_seg.avg):
            msg += (pf % (class_name[i], iou)+'\n')

    msg += f'\n\n \
            Segment:    Acc({segment_result[0]:.3f})    mIOU({segment_result[2]:.3f})\n\
            Time: inference({t:.4f}s/frame)'
    print(msg)
    if(logger):
        logger.info(msg)
    write_log(results_file, msg)

    
    
    return segment_result, losses.avg, t
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, 
                        default='hyp/hyp.yaml', 
                        help='hyperparameter path')
    parser.add_argument('--cfg', type=str, default='UNext', 
                                            help='model yaml path')
    parser.add_argument('--data', type=str, default='data/full.yaml', 
                                            help='dataset yaml path')
    parser.add_argument('--logDir', type=str, default='runs/test',
                            help='log directory')
    parser.add_argument('--img_size', nargs='+', type=int, default=[2048, 2048], 
                            help='[train, test] image sizes')
    parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default='weights/epoch-370.pth', 
                                                        help='model.pth path(s)')
    parser.add_argument('--batch_size', type=int, default=1, 
                            help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=0, 
                            help='maximum number of dataloader workers')
    return parser.parse_args()



if __name__ == '__main__':

    args = parse_args()

    device = select_device(args.device, batch_size=args.batch_size)


    # Hyperparameter
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Get class and class number
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    class_name = data_dict['class_name']
    hyp.update({'num_classes':len(class_name)})
    num_classes = hyp['num_classes']
    logger.info(f"{colorstr('class: ')}{class_name}")
    class_color = data_color(class_name)
    
    # Directories
    args.save_dir = Path(increment_path(Path(args.logDir)))  # increment run
    results_file = args.save_dir / 'results.txt'
    args.save_dir.mkdir(parents=True, exist_ok=True)


    # build up model
    print("begin to build up model...")
    model = build_model(args.cfg, num_classes).to(device)

    # loss function 
    criterion = Loss(hyp, device)

    # load weights
    model_dict = model.state_dict()
    checkpoint_file = args.weights
    print("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
    checkpoint_dict = checkpoint['state_dict']
    # checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    print("=> loaded checkpoint '{}' ".format(checkpoint_file))

    model = model.to(device)
    model.gr = 1.0
    model.nc = hyp['num_classes']
    print('bulid model finished')

    epoch = checkpoint['epoch'] #special for test
    # Save run settings

        # Data loading
    print("begin to load data")
    normalize = {'mean':[0.485, 0.456, 0.406], 
                 'std':[0.229, 0.224, 0.225]}
    valid_loader, valid_dataset = create_dataloader(args, hyp, data_dict,\
                                                args.batch_size, normalize, \
                                                is_train=False, shuffle=False)
    print('load data finished')

    with open(args.save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(args.save_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)
    
    test(epoch, args, hyp, valid_loader, model,
                                        criterion,args.save_dir,results_file,
                                        class_name,class_color,logger, device)

    print("test finish")