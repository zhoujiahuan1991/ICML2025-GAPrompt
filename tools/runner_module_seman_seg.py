import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import os
import ipdb
import numpy as np
from datasets import data_transforms
import cv2
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
from utils.config import cfg_from_yaml_file
from tqdm import tqdm
import sys
from utils import provider
from datasets.S3DISDataset import S3DISDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def run_net(args, config, train_writer=None, val_writer=None):

    logger = get_logger(args.log_name)
    # build dataset
    # (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)
    # (_, test_dataloader) = builder.dataset_builder(args, config.dataset.val)

    TRAIN_DATASET = S3DISDataset(split='train', data_root=config.root, num_point=config.npoints, test_area=5)
    train_dataloader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=config.total_bs, shuffle=True, num_workers=args.num_workers, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    TEST_DATASET = S3DISDataset(split='test', data_root=config.root, num_point=config.npoints, test_area=5)
    test_dataloader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=config.total_bs, shuffle=False, num_workers=args.num_workers)

    num_classes = 13

    # build model
    classifier = builder.model_builder(config.model)
    criterion = classifier.get_loss
    classifier.apply(inplace_relu)
    # print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    start_epoch = 0
    

    if args.ckpts is not None:
        classifier.load_model_from_ckpt(args.ckpts)
    else:
        print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        classifier.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        classifier = nn.DataParallel(classifier).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(classifier, config)

    print_log("Require gradient parameters: ", logger = logger)
    for name, param in classifier.named_parameters():
        if 'label_conv' in name or 'propagation_0' in name or 'point_prompt' in name or 'shift_net' in name or 'shape_feature_mlp' in name or 'adapter' in name  or 'cls_pos' in name or 'cls_token' in name or 'seg_head' in name or "prompt_embeddings" in name or 'prompt_cor' in name or 'out_transform' in name: 
            print_log(name, logger = logger)
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    from utils.misc import summary_parameters
    summary_parameters(classifier, logger=logger)

    best_acc = 0
    global_epoch = 0
    best_inctance_avg_iou = 0
    best_iou = 0
    time_sec_tot = 0.
    epoch_start_time = time.time()

    # test_metrics = validate(logger, num_part, classifier, test_dataloader, num_classes, config)
    import shutil
    shutil.copy('models/prompt_MAE_sem_segment.py', str(args.experiment_path))

    test_metrics = validate(logger, classifier, test_dataloader, num_classes, config, weights, criterion)
    # training
    classifier.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        classifier.train()

        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        classifier.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        mean_correct = []
        print_log('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, config.max_epoch), logger = logger)
        '''Adjust learning rate and BN momentum'''
        classifier = classifier.train()
        loss_batch = []
        num_iter = 0
        '''learning one epoch'''
        for batch_idx, (points, target) in enumerate(tqdm(train_dataloader)):
            num_iter += 1
            n_itr = epoch * n_batches + batch_idx
            data_time.update(time.time() - batch_start_time)
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_pointcloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)

            points, target = points.float().cuda(),  target.long().cuda()

            seg_pred = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            acc = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(acc.item() / (config.total_bs * config.npoints))
            loss = criterion(seg_pred, target, weights)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.detach().cpu())

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                loss_batch.append(loss.detach().cpu())
                classifier.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            # break

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)


        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        time_sec_tot += epoch_time
        time_sec_avg = time_sec_tot / (epoch - start_epoch + 1)

        train_instance_acc = np.mean(mean_correct)
        loss1 = np.mean(loss_batch)
        print_log('Train accuracy is: %.5f' % (train_instance_acc*100), logger = logger)
        print_log('Train loss: %.5f' % loss1, logger = logger)
        print_log('lr: %.6f' % optimizer.param_groups[0]['lr'], logger = logger)


        if epoch % args.val_freq == 0:
            test_metrics = validate(logger, classifier, test_dataloader, num_classes, config, weights, criterion)
            
            # Save ckeckpoints
            if (test_metrics['mIoU'] >= best_inctance_avg_iou):
                best_metrics = test_metrics
                builder.save_checkpoint(classifier, optimizer, epoch, test_metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
            if test_metrics['mIoU'] > best_inctance_avg_iou:
                best_inctance_avg_iou = test_metrics['mIoU']
            print_log('Best accuracy is: %.5f ' % (best_acc*100), logger = logger)
            print_log('Best inctance avg mIOU is: %.5f ' % (best_inctance_avg_iou*100), logger = logger)

        builder.save_checkpoint(classifier, optimizer, epoch, test_metrics, best_metrics, 'ckpt-last', args, logger = logger) 

        print_log('Epoch %d test Accuracy: %f   Inctance avg mIOU: %f' % (epoch + 1, test_metrics['accuracy']*100, test_metrics['mIoU']*100), logger = logger)
        global_epoch += 1



def validate(logger, classifier, testDataLoader, num_classes, config, weights, criterion):
    NUM_CLASSES = num_classes
    NUM_POINT = config.npoints
    BATCH_SIZE = config.total_bs
    test_metrics = {}
    with torch.no_grad():
        num_batches = len(testDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        classifier = classifier.eval()

        for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            # points = points.transpose(2, 1)

            seg_pred = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, weights)
            loss_sum += loss
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
            labelweights += tmp

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float)))    # np.float -> float
        print_log('eval mean loss: %f' % (loss_sum / float(num_batches)), logger=logger)
        print_log('[mIoU] eval point avg class IoU: %f' % (mIoU * 100.0), logger=logger)
        print_log('[OA] eval point accuracy: %f' % (total_correct / float(total_seen) * 100.0), logger=logger)
        print_log('[mAcc] eval point avg class acc: %f' % (np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float))) * 100.0), logger=logger)

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                total_correct_class[l] / float(total_iou_deno_class[l]) * 100.0)

        print_log(iou_per_class_str, logger=logger)
        print_log('Eval mean loss: %f' % (loss_sum / num_batches), logger=logger)
        print_log('Eval accuracy: %f' % (total_correct / float(total_seen) * 100.0), logger=logger)

        # test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['accuracy'] = float(np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float))))
        test_metrics['class_avg_accuracy'] = np.mean(np.array(total_correct_class) / np.array(total_iou_deno_class, dtype=float))
        test_metrics['mIoU'] = mIoU

    return test_metrics