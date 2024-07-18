"""Various utilities for neural networks."""
import math
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# from . import logger

INITIAL_LOG_LOSS_SCALE = 20.0


def convert_module_to_f16(l):
    """Convert primitive modules to float16."""
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """Convert primitive modules to float32, undoing convert_module_to_f16()."""
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def make_master_params(param_groups_and_shapes):
    """Copy model parameters into a (differently-shaped) list of full-precision parameters."""
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors([param.detach().float() for (_, param) in param_group]).view(
                shape
            )
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """Copy the gradients from the model parameters into the master parameters from
    make_master_params()."""
    for master_param, (param_group, shape) in zip(master_params, param_groups_and_shapes):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """Copy the master parameter data back into the model parameters."""
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(model, param_groups_and_shapes, master_params, use_fp16):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [(name, state_dict[name]) for name, _ in model.named_parameters()]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)


# class MixedPrecisionTrainer:
#     def __init__(
#         self,
#         *,
#         model,
#         use_fp16=False,
#         fp16_scale_growth=1e-3,
#         initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
#     ):
#         self.model = model
#         self.use_fp16 = use_fp16
#         self.fp16_scale_growth = fp16_scale_growth

#         self.model_params = list(self.model.parameters())
#         self.master_params = self.model_params
#         self.param_groups_and_shapes = None
#         self.lg_loss_scale = initial_lg_loss_scale

#         if self.use_fp16:
#             self.param_groups_and_shapes = get_param_groups_and_shapes(
#                 self.model.named_parameters()
#             )
#             self.master_params = make_master_params(self.param_groups_and_shapes)
#             self.model.convert_to_fp16()

#     def zero_grad(self):
#         zero_grad(self.model_params)

#     def backward(self, loss: th.Tensor):
#         if self.use_fp16:
#             loss_scale = 2**self.lg_loss_scale
#             (loss * loss_scale).backward()
#         else:
#             loss.backward()

#     def optimize(self, opt: th.optim.Optimizer):
#         if self.use_fp16:
#             return self._optimize_fp16(opt)
#         else:
#             return self._optimize_normal(opt)

#     def _optimize_fp16(self, opt: th.optim.Optimizer):
#         logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
#         model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
#         grad_norm, param_norm = self._compute_norms(grad_scale=2**self.lg_loss_scale)
#         if check_overflow(grad_norm):
#             self.lg_loss_scale -= 1
#             logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
#             zero_master_grads(self.master_params)
#             return False

#         logger.logkv_mean("grad_norm", grad_norm)
#         logger.logkv_mean("param_norm", param_norm)

#         for p in self.master_params:
#             p.grad.mul_(1.0 / (2**self.lg_loss_scale))
#         opt.step()
#         zero_master_grads(self.master_params)
#         master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
#         self.lg_loss_scale += self.fp16_scale_growth
#         return True

#     def _optimize_normal(self, opt: th.optim.Optimizer):
#         grad_norm, param_norm = self._compute_norms()
#         logger.logkv_mean("grad_norm", grad_norm)
#         logger.logkv_mean("param_norm", param_norm)
#         opt.step()
#         return True

#     def _compute_norms(self, grad_scale=1.0):
#         grad_norm = 0.0
#         param_norm = 0.0
#         for p in self.master_params:
#             with th.no_grad():
#                 param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
#                 if p.grad is not None:
#                     grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
#         return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

#     def master_params_to_state_dict(self, master_params):
#         return master_params_to_state_dict(
#             self.model, self.param_groups_and_shapes, master_params, self.use_fp16
#         )

#     def state_dict_to_master_params(self, state_dict):
#         return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """Update target parameters to be closer to those of source parameters using an exponential
    moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """Scale the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period)
        * th.arange(start=0, end=half, dtype=th.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """Evaluate a function without caching intermediate activations, allowing for reduced memory at
    the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads