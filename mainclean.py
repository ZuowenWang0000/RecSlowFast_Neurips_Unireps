#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:04:57 2019
Modified on Wed 15.Feb.2023 by Zuowen WANG @ institute of neuroinformatics, UZH/ETH
"""

import os, time, torch, mlflow, matplotlib, json, random
import multiprocessing as mp
from torchvision.transforms import Compose as transcompose
import torch.nn.parallel
import torch.optim
import numpy as np
from utils.dataset import DataSetSegVid
from models.hgrucleanSEG import hConvGRU, hConvGRUFastSlow
from models.convlstm import ConvLSTMFastSlow
from models.FFnet import FFConvNet_6L, FFConvNet_8L, FFConvNet_6L_skip
from utils.transforms import Augmentation, Stack, ToTorchFormatTensor
from utils.misc_functions import AverageMeter, FocalLoss, acc_scores, save_checkpoint, log_mlflow_metrics
from statistics import mean
from utils.opts import parser

matplotlib.use('Agg')

torch.backends.cudnn.benchmark = True

global best_prec1
best_prec1 = 0
args = parser.parse_args()
transform_list = transcompose([Augmentation(), Stack(), ToTorchFormatTensor(div=True)])
test_transform_list = transcompose([Stack(), ToTorchFormatTensor(div=True)])

pf_root = args.pf_root
img_path = os.path.join(pf_root, 'imgs/1')

with open(os.path.join(pf_root, 'dataset_config.json'), 'r') as f:
    config = json.load(f)
    total_num_videos = config['n_videos']
    vid_len = config['n_frames']


print("Loading training dataset and val dataset")
train_dataset = DataSetSegVid(img_path, os.path.join(pf_root,args.train_list), vid_len, transform=transform_list)
num_val_samples = int(len(train_dataset)*0.1)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset)-num_val_samples, num_val_samples]
                        ,generator = torch.Generator().manual_seed(42))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size, shuffle=True, num_workers=min(mp.cpu_count()//2,8),
                                           pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=args.batch_size, shuffle=False, num_workers=min(mp.cpu_count()//2,8),
                                         pin_memory=False, drop_last=False)

print("Loading test dataset")
test_loader = torch.utils.data.DataLoader(DataSetSegVid(img_path, os.path.join(pf_root, args.test_list), vid_len,transform=test_transform_list),
                                         batch_size=args.batch_size, shuffle=False, num_workers=min(mp.cpu_count()//2,8),
                                         pin_memory=False, drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def validate(loader, model, criterion, epoch, logiters=None, test=False):
    batch_timev = AverageMeter()
    lossesv = AverageMeter()
    iouv = AverageMeter()
    top1v = AverageMeter()
    precisionv = AverageMeter()
    recallv = AverageMeter()
    f1scorev = AverageMeter()

    model.eval()
    valortest = 'val' if not test else 'test'
    end = time.time()
    with torch.no_grad():
        for i, (imgs, target) in enumerate(loader):
            vid_loss = 0
            if args.random_init:
                # initialize hidden states from an gaussian distribution
                states = ['rand']*timesteps
            else:
                states = [None]*timesteps

            for t in range(len(target)):
                target_t = target[t].cuda()
                target_t = (target_t > 0.2).squeeze().long()
                imgs_t = imgs[t].cuda()

                rec_steps = args.slowsteps if t==0 else args.faststeps
                if args.stateful:
                    output, _, loss, states = model.forward(imgs_t, 0, 0, target_t, criterion, states[-1], rec_steps=rec_steps)
                else: 
                    output, _, loss, states = model.forward(imgs_t, 0, 0, target_t, criterion, None, rec_steps=rec_steps)
                
                loss = loss.mean()
                vid_loss += loss.item()
                iou, prec1, preci, rec, f1s = acc_scores(target_t, output.data, iou_th = args.iou_th) #default th is 0.5
                
                bs = target_t.shape[0]
                lossesv.update(loss.data.item(), bs)
                iouv.update(iou.data.item(), bs)
                top1v.update(prec1.item(), bs)
                precisionv.update(preci.item(), bs)
                recallv.update(rec.item(), bs)
                f1scorev.update(f1s.item(), bs)

                if i == 0:
                    # save ground truth and prediction
                    gt = target_t[0].data.cpu().unsqueeze(0).numpy()
                    pred = output[0].data.cpu().topk(1, 0, True, True)[1].numpy()
                    imgs_t = imgs_t[0].data.cpu().numpy()
                    margin_width = 7
                    margin = np.full(shape=(1, gt.shape[1], margin_width), fill_value=0.4)
                    concat_image = np.concatenate((imgs_t, margin, gt, margin, pred), axis=2).squeeze()
                    
                    if t == 0:
                        total_image = concat_image
                    else:
                        margin_h = np.full(shape=(margin_width, gt.shape[2]*3+margin_width*2), fill_value=0.4)
                        total_image = np.concatenate((total_image, margin_h, concat_image), axis=0)

            if i == 0: mlflow.log_image((total_image*255.).astype(np.uint8), f"imgs/{valortest}.ep{epoch}_gt_pred.jpg")
                    
            batch_timev.update(time.time() - end)
            end = time.time()
            vid_loss = vid_loss / len(target)
            if (i % args.print_freq == 0 or (i == len(loader) - 1)) and logiters is None:
                print_string = '{valortest}: [{step}/{total_step}]\t Time: {batch_time.avg:.3f}\t Loss: {loss.val:.8f} ({loss.avg: .8f})\t'\
                               'IOU avg:{iouv.avg:.8f} Bal_acc: {balacc:.8f} preci: {preci.val:.5f} ({preci.avg:.5f}) rec: {rec.val:.5f}'\
                               '({rec.avg:.5f}) f1: {f1s.val:.5f} ({f1s.avg:.5f})'\
                               .format(valortest=valortest, step=i, total_step=len(loader), batch_time=batch_timev, loss=lossesv, 
                                       balacc=top1v.avg, preci=precisionv, rec=recallv, f1s=f1scorev, iouv=iouv)
                print(print_string)
                with open(results_folder + args.name + '.txt', 'a+') as log_file:
                    log_file.write(print_string + '\n')

            elif logiters is not None:
                if i > logiters:
                    break
    model.train()
    return iouv.avg, top1v.avg, precisionv.avg, recallv.avg, f1scorev.avg, lossesv.avg


def save_npz(epoch, log_dict, results_folder, savename='train'):
    with open(results_folder + savename + '.npz', 'wb') as f:
        np.savez(f, **log_dict)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    exp = mlflow.set_experiment(args.exp_name)
    with mlflow.start_run(run_name=args.run_name,experiment_id=exp.experiment_id) as run:
        run_id = run.info.run_id

        print(f"cuda device available? :{torch.cuda.is_available()}  device count:{torch.cuda.device_count()}  current device:{torch.cuda.current_device()}")
        results_folder = f'./results/{args.name}/'
        args.results_folder = results_folder
        if not args.debug:
            os.mkdir(results_folder)

        with open(os.path.join(results_folder,'config.json'), 'w') as fp:
            json.dump(args.__dict__, fp, sort_keys=True, indent=4)
        with open(os.path.join(results_folder,'train_script.py'), "w", encoding='utf-8') as f:
            f.write(open(__file__, 'r', encoding='utf-8').read())
        # Write all files in results folder to artifacts
        mlflow.log_artifact(os.path.join(results_folder,'config.json'))
        mlflow.log_artifact(os.path.join(results_folder,'train_script.py'))
        
        exp_logging = args.log
        jacobian_penalty = args.penalty
        timesteps = args.train_timesteps
        assert timesteps == vid_len
        fs = args.filt_size
        set_seed(args.seed)

        if args.model == 'hgru':
            print("Init model hgru ", args.algo, 'penalty: ', args.penalty, 'steps: ', timesteps)
            model = hConvGRU(timesteps=timesteps, filt_size=fs, num_iter=15, exp_name=args.name, jacobian_penalty=jacobian_penalty,
                            grad_method=args.algo)
        elif args.model == 'hgrufs':
            print("Init model hgru fastslow ", args.algo, 'penalty: ', args.penalty, 'slow steps: ', args.slowsteps, 'fast steps: ', args.faststeps)
            model = hConvGRUFastSlow(filt_size=fs, num_iter=15, exp_name=args.name, 
                            jacobian_penalty=jacobian_penalty, grad_method=args.algo)        
        elif args.model == 'clstmfs':
            print("Init model clstm fastslow", args.algo, 'penalty: ', args.penalty, 'slow steps: ', args.slowsteps, 'fast steps: ', args.faststeps)
            model = ConvLSTMFastSlow(filt_size=fs, num_iter=15, exp_name=args.name, jacobian_penalty=jacobian_penalty,
                            grad_method=args.algo)
        elif args.model == 'ff6':
            print("Init model feedforw ", args.algo)
            model = FFConvNet_6L(filt_size=fs)
        elif args.model == 'ff8':
            print("Init model feedforw ", args.algo)
            model = FFConvNet_8L(filt_size=fs)
        elif args.model == 'ff6skip':
            model = FFConvNet_6L_skip(filt_size=fs)
        else:
            print('Model not found')
        print(f"trainable parameters:{sum([p.numel() for p in model.parameters() if p.requires_grad])}")


        if args.parallel is True:
            model = torch.nn.DataParallel(model).to(device)
            print("Loading parallel finished on GPU count:", torch.cuda.device_count())
        else:
            model = model.to(device)
            print("Loading finished")

        criterion = FocalLoss(gamma=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.7)
        lr_init = args.lr

        val_log_dict = {'iou': [],'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': []}
        test_log_dict = {'iou': [],'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': []}
        train_log_dict = {'iou': [], 'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': [], 
            'jvpen': [], 'scaled_loss': [], 'vid_mean_jvpen': []}


        exp_loss = None
        scale = torch.Tensor([1.0]).to(device)
        best_f1v = 0

        mlflow.log_params(args.__dict__)
        for epoch in range(args.start_epoch, args.epochs):

            batch_time = AverageMeter() 
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            precision = AverageMeter()
            recall = AverageMeter()
            f1score = AverageMeter()
            iou_averager = AverageMeter()

            time_since_last = time.time()
            model.train()
            end = time.perf_counter()

            
            for i, (imgs, target) in enumerate(train_loader):

                data_time.update(time.perf_counter() - end)
                vid_loss = 0
                jv_penalty_acc = 0

                if args.random_init:
                    # initialize hidden states from an gaussian distribution
                    states = ['rand']*timesteps
                else:
                    states = [None]*timesteps
                for t in range(vid_len):
                    imgs_t = imgs[t].to(device)
                    target_t = target[t].to(device)
                    target_t = (target_t > 0.2).squeeze().long()

                    rec_steps = args.slowsteps if t==0 else args.faststeps
                    if args.stateful:
                        output, jv_penalty, loss, states = model.forward(imgs_t, epoch, i, target_t, criterion, states[-1], rec_steps=rec_steps)
                    else:
                        output, jv_penalty, loss, states = model.forward(imgs_t, epoch, i, target_t, criterion, None, rec_steps=rec_steps)
                    loss = loss.mean()

                    losses.update(loss.data.item(), 1)
                    jv_penalty = jv_penalty.mean()
                
                    train_log_dict['jvpen'].append(jv_penalty.item())
                    jv_penalty_acc += jv_penalty.item()
                    
                    if jacobian_penalty:
                        loss = loss + jv_penalty * 1e1
                    
                    vid_loss = vid_loss + loss
                    iou, acc, preci, rec, f1s = acc_scores(target_t[:], output.data[:], iou_th=args.iou_th)
                    
                    bs = imgs_t.shape[0]
                    top1.update(acc.item(), bs)
                    iou_averager.update(iou.item(), bs)
                    precision.update(preci.item(), bs)
                    recall.update(rec.item(), bs)
                    f1score.update(f1s.item(), bs)

                train_log_dict['vid_mean_jvpen'].append(jv_penalty_acc/vid_len)
                vid_loss = vid_loss / vid_len
                vid_loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                batch_time.update(time.perf_counter() - end)
                
                end = time.perf_counter()

                if i % (args.print_freq) == 0:
                    time_now = time.time()
                    print_string = 'Epoch: [{0}][{1}/{2}]  lr: {lr:g} '\
                    'Loss: {loss.val:.8f} ({loss.avg:.8f}) IOU avg: {iou.avg:.8f} bal_acc: {top1.val:.5f} '\
                    '({top1.avg:.5f}) preci: {preci.val:.5f} ({preci.avg:.5f}) rec: {rec.val:.5f} '\
                    '({rec.avg:.5f})  f1: {f1s.val:.5f} ({f1s.avg:.5f}) jvpen: {jpena:.12f} losscale:{losscale:.5f}'\
                    .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                            lossprint=mean(losses.history[-args.print_freq:]), lr=optimizer.param_groups[0]['lr'], iou=iou_averager,
                            top1=top1, timeiteravg=mean(batch_time.history[-args.print_freq:]),
                            timeprint=time_now - time_since_last, preci=precision, rec=recall,
                            f1s=f1score, jpena=jv_penalty.item(), losscale=scale.item())

                    print(print_string)
                    time_since_last = time_now
                    with open(results_folder + args.name + '.txt', 'a+') as log_file:
                        log_file.write(print_string + '\n')

            train_log_dict['loss'].extend(losses.history)
            train_log_dict['balacc'].extend(top1.history)
            train_log_dict['precision'].extend(precision.history)
            train_log_dict['recall'].extend(recall.history)
            train_log_dict['f1score'].extend(f1score.history)
            train_log_dict['iou'].extend(iou_averager.history)
            save_npz(epoch, train_log_dict, results_folder, 'train')
            save_npz(epoch, val_log_dict, results_folder, 'val')
            save_npz(epoch, test_log_dict, results_folder, 'test')
            log_mlflow_metrics(mlflow, 'tr', epoch, iou_averager.avg, top1.avg, precision.avg, recall.avg, f1score.avg, losses.avg)


            if (epoch % args.val_freq == 0) or (epoch == args.epochs - 1):
                iouv, accv, precv, recv, f1sv, losv = validate(val_loader, model, criterion, epoch, logiters=3)
                print(f'val f:{f1sv}, iouv:{iouv}')
                val_log_dict['loss'].append(losv)
                val_log_dict['iou'].append(iouv)
                val_log_dict['balacc'].append(accv)
                val_log_dict['precision'].append(precv)
                val_log_dict['recall'].append(recv)
                val_log_dict['f1score'].append(f1sv)

                log_mlflow_metrics(mlflow, 'val', epoch, iouv, accv, precv, recv, f1sv, losv)
            if (epoch % args.val_freq == 0 and f1sv > best_f1v) or (epoch % args.test_freq == 0):
                if f1sv > best_f1v: 
                    best_f1v = f1sv
                    # encountered a new best on validation, save the model
                    save_checkpoint(mlflow, model, optimizer, epoch, losv, iouv, losses.avg, iou_averager.avg)
                
                # we test the model anyways whether the val is improved or test frequency comes
                iout, acct, prect, rect, f1st, lost = validate(test_loader, model, criterion, epoch, test=True)
                log_mlflow_metrics(mlflow, 'test', epoch, iout, acct, prect, rect, f1st, lost)
