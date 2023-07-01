import torch
import torch.nn as nn
import onnx
from . import funcs
from . import tensor
import onnx.helper as helper
import numpy as np
from mmdet3d.ops import spconv as spconv
from .quantize import QuantAdd

avoid_reuse_container = []
obj_to_tensor_id = {}
nodes = []
initializers = []
enable_trace = False
inverse_indices = False

def register_node(fn):

    fnnames   = fn.split(".")
    fn_module = eval(".".join(fnnames[:-1]))
    fn_name   = fnnames[-1]
    oldfn     = getattr(fn_module, fn_name)
    
    def make_hook(bind_fn):

        ilayer = 0
        def internal_forward(self, *args):
            global enable_trace

            if not enable_trace:
                return oldfn(self, *args)

            global avoid_reuse_container
            nonlocal ilayer

            # Use the enable_trace flag to avoid internal trace calls
            enable_trace = False
            y = oldfn(self, *args)
            bind_fn(self, ilayer, y, *args)
            enable_trace = True

            avoid_reuse_container.extend(list(args) + [y]) 
            ilayer += 1
            return y

        setattr(fn_module, fn_name, internal_forward)
    return make_hook


@register_node("torch.nn.ReLU.forward")
def symbolic_relu(self, ilayer, y, x):
    register_tensor(y)
    print(f"   --> ReLU{ilayer} -> Input {get_tensor_id(x)}, Output {get_tensor_id(y)}")

    nodes.append(
        helper.make_node(
            "Relu", [get_tensor_id(x)], [get_tensor_id(y)], f"relu{ilayer}"
        )
    )


@register_node("QuantAdd.forward")
def symbolic_add_quant(self, ilayer, y, a, b):
    register_tensor(y)
    print(f"   --> QuantAdd{ilayer} -> Input {get_tensor_id(a)} + {get_tensor_id(b)}, Output {get_tensor_id(y)}")

    nodes.append(
        helper.make_node(
            "Add", [get_tensor_id(a), get_tensor_id(b)], [get_tensor_id(y)], f"add{ilayer}",
            input0_dynamic_range = self._input_quantizer.amax.cpu().item(),
            input1_dynamic_range = self._input_quantizer.amax.cpu().item(),
            precision            = "fp16" if hasattr(self, "precision") is None else self.precision,
            output_precision     = "fp16" if hasattr(self, "output_precision") is None else self.output_precision
        )
    )


@register_node("torch.Tensor.__add__")
def symbolic_add(a, ilayer, y, b):
    register_tensor(y)
    print(f"   --> Add{ilayer} -> Input {get_tensor_id(a)} + {get_tensor_id(b)}, Output {get_tensor_id(y)}")

    nodes.append(
        helper.make_node(
            "Add", [get_tensor_id(a), get_tensor_id(b)], [get_tensor_id(y)], f"add{ilayer}"
        )
    )


@register_node("torch.Tensor.reshape")
def node_view(self, ilayer, y, *dims):
    register_tensor(y)
    print(f"   --> Reshape{ilayer}[{dims}] -> Input {get_tensor_id(self)}, Output {get_tensor_id(y)}")

    nodes.append(
        helper.make_node(
            "Reshape", [get_tensor_id(self)], [get_tensor_id(y)], f"reshape{ilayer}",
            dims = dims
        )
    )


@register_node("torch.Tensor.permute")
def node_permute(self, ilayer, y, *dims):
    register_tensor(y)
    print(f"   --> Permute{ilayer}[{dims}][{list(y.shape)}] -> Input {get_tensor_id(self)}, Output {get_tensor_id(y)}")

    nodes.append(
        helper.make_node(
            "Transpose", [get_tensor_id(self)], [get_tensor_id(y)], f"transpose{ilayer}",
            dims = dims
        )
    )


def printtensor(x):
    x = x.features
    print(x.min().item(), x.max().item(), x.std().item(), x.mean().item())


def make_model_forward_hook(self, inverse_indices=False):
    def impl(voxel_features, coors, batch_size, **kwargs):
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        if inverse_indices:
            N, C, Z, Y, X    = spatial_features.shape
            spatial_features = spatial_features.permute(0, 1, 2, 4, 3)
            spatial_features = spatial_features.reshape(N, C * Z, X, Y)
        else:
            N, C, X, Y, Z    = spatial_features.shape
            spatial_features = spatial_features.permute(0, 1, 4, 2, 3)
            spatial_features = spatial_features.reshape(N, C * Z, X, Y)
        print('onnx output shape:', spatial_features.shape)
        return spatial_features
    return impl


def append_initializer(value, name):
    initializers.append(
        helper.make_tensor(
            name=name,
            data_type=helper.TensorProto.DataType.FLOAT16,
            dims=list(value.shape),
            vals=value.cpu().data.numpy().astype(np.float16).tobytes(),
            raw=True
        )
    )
    return name


def __obj_to_id(obj):
    idd = id(obj)
    if isinstance(obj, spconv.SparseConvTensor):
        idd = id(obj.features)
    return idd


def set_obj_idd_assame(a_already_has_idd, b_no_idd):
    global obj_to_tensor_id
    aidd = __obj_to_id(a_already_has_idd)
    bidd = __obj_to_id(b_no_idd)
    
    assert aidd in obj_to_tensor_id, "A is not in tensor map"
    assert bidd not in obj_to_tensor_id, "B is already in tensor map"
    obj_to_tensor_id[bidd] = obj_to_tensor_id[aidd]


def register_tensor(obj):
    global obj_to_tensor_id
    obj_to_tensor_id[__obj_to_id(obj)] = str(len(obj_to_tensor_id))


def get_tensor_id(obj):
    idd = __obj_to_id(obj)
    assert idd in obj_to_tensor_id, "ops!!!😮 Cannot find the tensorid of this object. this means that some operators are not being traced. You need to confirm it."
    return obj_to_tensor_id[idd]


def inverse_model(model : nn.Module):
    # change index xyz to zyx
    model.sparse_shape = model.sparse_shape[::-1]
    for name, module in model.named_modules():
        if isinstance(module, spconv.conv.SparseConvolution):
            # (xyz) I, O
            module.weight.data    = module.weight.data.permute(2, 1, 0, 3, 4).contiguous()
            module.padding        = module.padding[::-1]
            module.stride         = module.stride[::-1]
            module.dilation       = module.dilation[::-1]
            module.kernel_size    = module.kernel_size[::-1]
            module.output_padding = module.output_padding[::-1]


def inference_and_save_tensor(model : nn.Module, voxels, coors, batch_size, inverse, save_tensor):
    # process model weight/stride/padding/output_padding/dilation etc...
    if inverse:
        coors = coors[:, [0, 3, 2, 1]]
        inverse_model(model)

    spatial_shape = model.sparse_shape
    model.forward = make_model_forward_hook(model, inverse)

    print("> Do inference...")
    with torch.no_grad():
        y = model(voxels, coors, batch_size)

    print("> Do save tensor, The purpose of this operation is to verify the inference result of C++")
    print(f"   --> Save inference input voxels to {save_tensor}.voxels, voxels.shape = {voxels.shape}")
    tensor.save(voxels, f"{save_tensor}.voxels")

    print(f"   --> Save inference input coors to {save_tensor}.coors, coors.shape = {coors.shape}")
    tensor.save(coors,  f"{save_tensor}.coors")

    print(f"   --> Save inference output to {save_tensor}.dense, output.shape = {y.shape}")
    tensor.save(y,      f"{save_tensor}.dense")
    
    print(f"   --> Save spatial_shape is {spatial_shape}, batch size is {batch_size}")


def export_onnx(model : nn.Module, voxels, coors, batch_size, inverse, save_onnx):

    global avoid_reuse_container, obj_to_tensor_id, nodes, initializers, enable_trace, inverse_indices
    avoid_reuse_container = []
    obj_to_tensor_id      = {}
    nodes                 = []
    initializers          = []
    inverse_indices       = inverse
    spatial_shape         = model.sparse_shape

    if inverse:
        spatial_shape = spatial_shape[::-1]
        coors         = coors[:, [0, 3, 2, 1]]
        inverse_model(model)

    for i, layers in enumerate(model.encoder_layers):
        if len(layers) == 1:
            m0 = layers[0]

            # @!!!! Warning~  the first subm layer's indice_key is subm1
            m0.indice_key = f"subm{i+1}"

        elif len(layers) == 3:
            m0, m1, m2 = layers[0], layers[1], layers[2]
            
            # @!!!! Warning~  the first subm layer's indice_key is subm1
            m0.indice_key = f"subm{i+1}"
            m1.indice_key = f"subm{i+1}"
            m2.indice_key = f"subm{i+1}"

    model.forward = make_model_forward_hook(model, inverse)

    print("Tracing model inference...")
    print("> Do inference...")
    with torch.no_grad():
        register_tensor(voxels)
        enable_trace = True
        y            = model(voxels, coors, batch_size)
        enable_trace = False

    print("Tracing done!")
    inputs = [
        helper.make_value_info(
            name="0",
            type_proto=helper.make_tensor_type_proto(
                elem_type=helper.TensorProto.DataType.FLOAT16,
                shape=voxels.size()
            )
        )
    ]

    outputs = [
        helper.make_value_info(
            name=get_tensor_id(y),
            type_proto=helper.make_tensor_type_proto(
                elem_type=helper.TensorProto.DataType.FLOAT16,
                shape=y.size()
            )
        )
    ]

    graph = helper.make_graph(
        name="scn",
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializer=initializers
    )

    opset = [
        helper.make_operatorsetid("ai.onnx", 11)
    ]

    model = helper.make_model(graph, opset_imports=opset, producer_name="pytorch", producer_version="1.9")
    onnx.save_model(model, save_onnx)
    print(f"🚀🚀🚀 The onnx export is completed. ONNX save as {save_onnx} 😄😄😄, Have a nice day 🌹🌹🌹")

    # clean memory
    avoid_reuse_container = []
    obj_to_tensor_id = {}
    nodes = []
    initializers = []