import cv2
import yaml
import math
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter

import torch
import torch.optim
import torch.utils.data
from torch.cuda import amp
import torch.backends.cudnn


from utils.general import non_max_suppression, get_bounding_boxes, increment_path,\
                     set_logging, colorstr, write_log, data_color, AverageMeter
from utils.datasets import create_dataloader
from utils.plot import plot_BBox
from utils.loss import Loss
from utils.torch_utils import select_device
from models.model import build_model, get_optimizer
from test import test

logger = logging.getLogger(__name__)

def main(args, hyp, device, writer):
    begin_epoch, global_steps, best_fitness, fi = 1, 0, 0.0, 1.0

    # Directories
    save_dir, maxEpochs = Path(args.save_dir), args.epochs
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    results_file = save_dir / 'results.txt'

    last = wdir / f'last.pth'
    best = wdir / f'best.pth'

    # Get class and class number
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    class_name = data_dict['class_name']
    hyp.update({'num_classes':len(class_name)})
    num_classes = hyp['num_classes']
    logger.info(f"{colorstr('class: ')}{class_name}")
    class_color = data_color(class_name)
    
    # Save run settings(hyp, args)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # build up model
    print("begin to build up model...")
    model = build_model(args.cfg, num_classes).to(device)

    # loss function 
    criterion = Loss(hyp, device)

    # Optimizer
    optimizer = get_optimizer(hyp, model)                               

    # resume 
    if(args.resume):
        checkpoint = torch.load(args.resume)
        begin_epoch += checkpoint['epoch']
        global_steps = checkpoint['global_steps']+1
        best_fitness = checkpoint['best_fitness']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        msg = f'{colorstr("=> loaded checkpoint")} "{args.resume}"(epoch {begin_epoch})'
        logger.info(msg)
        write_log(results_file, msg)

    # Data loading
    print("begin to load data")
    normalize = {'mean':[0.485, 0.456, 0.406], 
                 'std':[0.229, 0.224, 0.225]}
    
    train_loader, train_dataset = create_dataloader(args, hyp, data_dict, \
                                                args.batch_size, normalize)
    num_batch = len(train_loader)
    
    valid_loader, valid_dataset = create_dataloader(args, hyp, data_dict,\
                                                args.batch_size, normalize, \
                                                    is_train=False, shuffle=False)

    print('load data finished')

    lf = lambda x: ((1 + math.cos(x * math.pi / maxEpochs)) / 2) * \
                   (1 - hyp['lrf']) + hyp['lrf']  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    model.gr = 1.0
    model.nc = hyp['num_classes']

    # training
    num_warmup = max(round(hyp['warmup_epochs'] * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    print(colorstr('=> start training...'))
    for epoch in range(begin_epoch, maxEpochs+1):

        model.train()
        start = time.time()
        for i, (input, target, paths, shapes) in enumerate(train_loader):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            num_iter = i + num_batch * (epoch - 1)

            if num_iter < num_warmup:
                # warm up
                lf = lambda x: ((1 + math.cos(x * math.pi / maxEpochs)) / 2)* \
                            (1 - hyp['lrf']) + hyp['lrf']  # cosine
                xi = [0, num_warmup]
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  
                # # iou loss ratio (obj_loss = 1.0 or iou)
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, 
                    # all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(num_iter, xi, [hyp['warmup_biase_lr'] \
                                        if j == 2 else 0.0, x['initial_lr'] *\
                                                                lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(num_iter, xi, 
                                                [hyp['warmup_momentum'], 
                                                    hyp['momentum']])
            

            data_time.update(time.time() - start)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # Forward
            with amp.autocast(enabled=device.type != 'cpu'):
                outputs = model(input)
                loss = criterion(outputs, target)

            # compute gradient and do update step
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
            #                                  target.detach().cpu().numpy())
            # acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - start)
            if i % 10 == 0:
                msg = f'Epoch: [{epoch}][{i}/{len(train_loader)}] '+\
                        f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)  '+\
                        f'peed {input.size(0)/batch_time.val:.1f} samples/s  '+\
                        f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)  '+\
                        f'Loss {losses.val:.5f} ({losses.avg:.5f})  '
                logger.info(msg)
                # Write 
                # write_log(results_file, msg)
                # validation result tensorboard
                writer.add_scalar('train_loss', losses.avg, global_steps)
                global_steps += 1

        lr_scheduler.step()      


        # evaluate on validation set
        if (epoch >= args.val_start and (epoch % args.val_freq == 0 
                                                    or epoch == maxEpochs)):

            segment_result, loss, t= test(epoch, args, hyp, valid_loader, model,
                                        criterion,save_dir,results_file,class_name, 
                                        class_color,logger, device)
            

            # validation result tensorboard
            writer.add_scalar('val_loss', loss, global_steps)
            # writer.add_scalar('Acc', segment_result[0], global_steps)
            # writer.add_scalar('IOU', segment_result[1], global_steps)
            # writer.add_scalar('mIOU', segment_result[2], global_steps)

        ckpt = {
            'epoch': epoch,
            'best_fitness': best_fitness,
            'global_steps':global_steps-1,
            'state_dict':  model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        # last
        torch.save(ckpt, last)
        # frequency
        if (epoch % args.val_freq == 0 or epoch == maxEpochs):
            savepath = wdir / f'epoch-{epoch}.pth'
            logger.info(f'{colorstr("=> saving checkpoint")} to {savepath}')
            torch.save(ckpt, savepath)
        # best
        if best_fitness == fi:
            logger.info(f'{colorstr("=> saving checkpoint")} to {savepath}')
            torch.save(ckpt, best)


        del ckpt

    torch.cuda.empty_cache()
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, 
                        default='hyp/hyp.yaml', 
                        help='hyperparameter path')
    parser.add_argument('--cfg', type=str, default='cfg/YOLOP_v7b3.yaml', 
                                            help='model yaml path')
    parser.add_argument('--data', type=str, default='data/full.yaml', 
                                            help='dataset yaml path')
    parser.add_argument('--resume', type=str, default='',
                            help='Resume the weight  runs/train/BddDataset/')
    parser.add_argument('--logDir', type=str, default='runs/train',
                            help='log directory')
    
    parser.add_argument('--batch_size', type=int, default=3, 
                            help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=0,
                         
                            help='maximum number of dataloader workers')
    parser.add_argument('--device', default='', 
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Max of epoch')
    
    parser.add_argument('--img_size', nargs='+', type=int, default=[1024, 1024], 
                            help='[train, test] image sizes')
    
    parser.add_argument('--Binary_thold', type=int,
                        default=230,
                        help='Binarization threshold')
    parser.add_argument('--nms_thold', type=float,
                        default=0.5,
                        help='non max suppression threshold')
    
    parser.add_argument('--val_start', type=int, default=20, 
                            help='start do validation')
    parser.add_argument('--val_freq', type=int, default=10, 
                            help='How many epochs do one time validation')
    
    # Cudnn related params
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,  
                                help='Use GPUs to speed up network training')
    parser.add_argument('--cudnn_deterministic', type=bool, default=False, 
                                help='only use deterministic convolution algorithms')
    parser.add_argument('--cudnn_enabled', type=bool, default=True,  
                                help='controls whether cuDNN is enabled')
    return parser.parse_args()




if __name__ == '__main__':

    args = parse_args()
    set_logging()

    # cudnn related setting
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = args.cudnn_deterministic

    device = select_device(args.device, batch_size=args.batch_size)

    # Hyperparameter
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    if(args.resume):
        args.save_dir = Path('./')
        for p in args.resume.split('/')[:-2]:
            args.save_dir= args.save_dir / p
    else:
        args.save_dir = increment_path(Path(args.logDir)) 
    print(args.save_dir)

    # Train
    logger.info(args)
    logger.info(f"{colorstr('tensorboard: ')}Start with 'tensorboard --logdir {args.logDir}'"+\
                                        ", view at http://localhost:6006/")
    writer = SummaryWriter(args.save_dir)  # Tensorboard

    main(args, hyp, device, writer)













# if __name__ == '__main__':

#     args = parse_args()
#     save_result = True

#     with open(args.data) as f:
#         data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
#     data_path = Path(data_dict['train'][0])

#     for img_path in data_path.glob('*.bmp'):
#         # 讀入圖片
#         img = cv2.imread(str(img_path))

#         # 取得未nms的bbox
#         boxes = get_bounding_boxes(img, args.Binary_thold, save_result)

#         # nms
#         boxes = non_max_suppression(boxes, args.nms_thold)

#         # 顯示圖片
#         plot_BBox(img, img_path.stem, boxes, save_result)