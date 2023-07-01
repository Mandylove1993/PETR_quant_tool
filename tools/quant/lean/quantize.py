import torch
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from typing import Callable
from absl import logging as quant_logging

from torch.nn.parameter import Parameter
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
import mmcv.cnn.bricks.wrappers
import torch.nn.functional as F

from torch import nn


class HardSimpleVFE(nn.Module):
    """Simple voxel feature encoder used in SECOND.

    It simply averages the values of points in a voxel.

    Args:
        num_features (int, optional): Number of features to use. Default: 4.
    """

    def __init__(self, num_features=4):
        super(HardSimpleVFE, self).__init__()
        self.num_features = num_features

    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, M, 3(4)). N is the number of voxels and M is the maximum
                number of points inside a single voxel.
            num_points (torch.Tensor): Number of points in each voxel,
                 shape (N, ).
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (N, 3(4))
        """
        points_mean = features[:, :, :self.num_features].sum(dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        return points_mean.contiguous()


class QuantConcat(torch.nn.Module):
    def __init__(self, quantization =True):
        super().__init__()

        if quantization:
            self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input_quantizer._calibrator._torch_hist = True
            self._fake_quant = True
        self.quantization = quantization

    def forward(self, x,  y):
        if self.quantization:
            return torch.cat([self._input_quantizer(x), self._input_quantizer(y)], dim=1)
        return torch.cat([x, y], dim=1)


class QuantAdd(torch.nn.Module):
    def __init__(self, quantization = True):
        super().__init__()
  
        if quantization:
            self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input_quantizer._calibrator._torch_hist = True
            self._fake_quant = True
        self.quantization = quantization
              
    def forward(self, input1, input2):
        if self.quantization:
            return torch.add(self._input_quantizer(input1), self._input_quantizer(input2))
        return torch.add(input1, input2)


def transfer_torch_to_quantization(nninstance : torch.nn.Module, quantmodule):

    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            quant_desc_input = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True)
            self.init_quantizer(quant_desc_input)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance


def replace_to_quantization_module(model : torch.nn.Module):

    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path      = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:  
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])
    recursive_and_replace_module(model)


class hook_fpn_forward:
    def __init__(self, obj):
        self.obj = obj


    def __call__(self, inputs):
        self = self.obj
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = self.quant_add(laterals[i - 1], F.interpolate(laterals[i],
                                                 **self.upsample_cfg))
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = self.quant_add(laterals[i - 1], F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg))

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) if i==0 else laterals[i] for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)



def quantize_neck(model_camera_neck):  
    replace_to_quantization_module(model_camera_neck)    
    model_camera_neck.quant_add = QuantAdd()
    model_camera_neck.forward      = hook_fpn_forward(model_camera_neck)
    


def quantize_head(model_head):
    replace_to_quantization_module(model_head)
    # replace_to_quantization_module(model_head.input_proj)    
    # replace_to_quantization_module(model_head.cls_branches)    
    # replace_to_quantization_module(model_head.reg_branches)    


def quantize_backbone(model_camera_backbone):
    replace_to_quantization_module(model_camera_backbone)


def calibrate_model(model : torch.nn.Module, dataloader, device, batch_processor_callback: Callable = None, num_batch=1):

    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(strict=False, **kwargs)

                    module._amax = module._amax.to(device)
        
    def collect_stats(model, data_loader, device, num_batch=200):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        iter_count = 0 
        for data in tqdm(data_loader, total=num_batch, desc="Collect stats for calibrating"):
            with torch.no_grad():
                # print(data['img'][0].data[0].shape)
                result = model(return_loss=False, **data)
            iter_count += 1
            if iter_count >num_batch:
                break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    collect_stats(model, dataloader, device, num_batch=num_batch)
    compute_amax(model, method="mse")


def print_quantizer_status(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print('TensorQuantizer name:{} disabled staus:{} module:{}'.format(name, module._disabled, module))


def have_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True
    return False


def set_quantizer_fast(module): 
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
             if isinstance(module._calibrator, calib.HistogramCalibrator):
                module._calibrator._torch_hist = True 


class disable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


def build_sensitivity_profile(model, data_loader_val, dataset_val, eval_model_callback : Callable = None):
    quant_layer_names = []
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            print('use quant layer:{}',name)
            module.disable()
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quant_layer_names:
                quant_layer_names.append(layer_name)
    for i, quant_layer in enumerate(quant_layer_names):
        print("Enable", quant_layer)
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
                module.enable()
                print(F"{name:40}: {module}")
        with torch.no_grad():
            eval_model_callback(model,data_loader_val, dataset_val) 
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
                module.disable()
                print(F"{name:40}: {module}")


def initialize():
    quant_logging.set_verbosity(quant_logging.ERROR)
    quant_desc_input = QuantDescriptor(calib_method="histogram")

    quant_modules._DEFAULT_QUANT_MAP.append(
        quant_modules._quant_entry(mmcv.cnn.bricks.wrappers, "ConvTranspose2d", quant_nn.QuantConvTranspose2d)
    )

    for item in quant_modules._DEFAULT_QUANT_MAP:
        item.replace_mod.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR) 
