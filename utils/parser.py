import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        default= 'cfgs/finetune_scan_objbg.yaml', 
        # ./cfgs/fewshot.yaml  ./cfgs/finetune_scan_hardest.yaml cfgs/finetune_scan_objbg.yaml cfgs/finetune_scan_objonly.yaml cfgs/finetune_modelnet.yaml
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=False,
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='point-mae-finetune-st', help = 'experiment name') 
    #femae-pointprompt-r-learningrate0.0005-per.aft.attn-point5
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default='pretrained_bases/mae_base.pth', help = 'test used ckpt path') 
    #pretrained_bases/mae_base.pth ./pretrained_bases/recon_base.pth  ./pretrained_bases/femae-epoch-300.pth
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help = 'vote acc')
    parser.add_argument(
        '--segmentation',
        action='store_true',
        default=False,
        help = 'point cloud segmentation task') 
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model', 
        action='store_true', 
        default=True, 
        help = 'finetune modelnet with pretrained weight')
    parser.add_argument(
        '--if_maxmean', 
        action='store_true', 
        default=False, 
        help = 'the method of pooling for block')
    parser.add_argument(
        '--propagate_cof', 
        type=float,
        help = 'in the propagate for block')
    parser.add_argument(
        '--center_cof', 
        type=float,
        help = 'in the forward for block')
    parser.add_argument(
        '--ad_cof', 
        type=float,
        help = 'in the forward for block')
    ###
    parser.add_argument(
        '--way', type=int, default=5)
    parser.add_argument(
        '--shot', type=int, default=-1)
    parser.add_argument(
        '--fold', type=int, default=9)
    
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        print('training from scratch')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

