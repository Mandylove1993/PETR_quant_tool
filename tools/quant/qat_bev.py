import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

import lean.quantize as quantize
import lean.funcs as funcs
from lean.train import qat_train

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, load_checkpoint, build_optimizer
from mmdet.datasets import replace_ImageToTensor

from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test
import torch.optim as optim
from torch.cuda import amp
# Additions
from mmcv.runner import  load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.cnn.utils.fuse_conv_bn import _fuse_conv_bn
from pytorch_quantization.nn.modules.quant_conv import QuantConv2d


def fuse_conv_bn(module):
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, QuantConv2d) or isinstance(child, nn.Conv2d): # or isinstance(child, QuantConvTranspose2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module


def load_model(cfg, checkpoint_path = None):
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if checkpoint_path != None:
        checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    return model, checkpoint


def quantize_net(model):  
    quantize.quantize_backbone(model.img_backbone)
    quantize.quantize_neck(model.img_neck)
    quantize.quantize_head(model.pts_bbox_head)
    return model
    

def test_model(cfg, args, model, checkpoint, data_loader, dataset):
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


def main():
    quantize.initialize()  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="FILE", 
                        default="projects/configs/petr/qat/petr_vovnet_gridmask_p4_800x320.py", 
                        help="config file")
    parser.add_argument("--ckpt", 
                        default='tools/quant/pth/petr_vov_p4_800x320.pth',
                        help="the checkpoint file to resume from")
    parser.add_argument("--calibrate_batch", type=int, default=200, help="calibrate batch")
    parser.add_argument("--seed", type=int, default=666, help="seed")
    parser.add_argument("--deterministic", type=bool, default=True, help="deterministic")
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where results will be saved')
    parser.add_argument('--test_int8_and_fp32', default=False, help='test int8 and fp32 or not')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        default='bbox',
        help='evaluation metrics, which depends on the dataset, e.g., "mAP",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()

    args.ptq_only          = True
    cfg                    = Config.fromfile(args.config)
    cfg.seed               = args.seed
    cfg.deterministic      = args.deterministic
    cfg.test_int8_and_fp32 = args.test_int8_and_fp32

    # set random seeds
    if cfg.seed is not None:
        print(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
                
    dataset_train  = build_dataset(cfg.data.train)
    dataset_test   = build_dataset(cfg.data.test)
    print('train nums:{} val nums:{}'.format(len(dataset_train), len(dataset_test)))   

    distributed =False
    
    data_loader_test =  build_dataloader(
            dataset_test,
            samples_per_gpu=1,  
            workers_per_gpu=4,  
            dist=distributed,
            shuffle=False,
        )
    
    print('Test DataLoader Info:', data_loader_test.batch_size, data_loader_test.num_workers)

    #Create Model
    model_fp32, checkpoint = load_model(cfg, checkpoint_path = args.ckpt)
    model_int8 = deepcopy(model_fp32)

    if cfg.test_int8_and_fp32:
        model_fp32 = fuse_conv_bn(model_fp32)
        model_fp32 = MMDataParallel(model_fp32, device_ids=[0])
        model_fp32.eval()
        print('############################## fp32 ##############################')
        test_model(cfg, args, model_fp32, checkpoint, data_loader_test, dataset_test)

    model_int8 = quantize_net(model_int8)
    model_int8 = fuse_conv_bn(model_int8)
    model_int8 = MMDataParallel(model_int8, device_ids=[0])
    model_int8.eval()

    ##Calibrate
    print("Start calibrate 🌹🌹🌹🌹🌹🌹  ")
    quantize.set_quantizer_fast(model_int8)
    quantize.calibrate_model(model_int8, data_loader_test, 0, None, args.calibrate_batch)
    
    save_ptq_path = 'tools/quant/pth/bev_ptq.pth'
    os.makedirs(os.path.dirname(save_ptq_path), exist_ok=True)
    print(f"Done due to ptq only! Save checkpoint to {save_ptq_path} 🌹🌹🌹🌹🌹🌹")
    torch.save(model_int8, save_ptq_path)

    if cfg.test_int8_and_fp32:
        print('############################## PTQ int8 ##############################')
        test_model(cfg, args, model_int8, checkpoint, data_loader_test, dataset_test)

    # model = train_model(model, data_loader_train, cfg, train_layers)
    # QAT train
    data_loader_train = [build_dataloader(
        dataset_train,
        samples_per_gpu=1,  
        workers_per_gpu=4,  
        dist=distributed,
        shuffle=False,
    )]

    cfg.run_dir = "tools/quant/qat/"
    qat_train(model_int8, dataset_train, data_loader_train, cfg, validate=True)

if __name__ == "__main__":
    main()
