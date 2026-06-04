import numpy as np
import torch
import torch.nn as nn

# def get_output_shape(input_shape, module):
#     """Takes an input shape and a module and returns the shape that
#     the module would return given a tensor of shape input_shape.
#     Assumes that the module's output shape only depends on the shape of
#     the input.

#     Args:
#         input_shape: any iterable that describes the shape. Shouldn't
#             include any batch size.
#         module: anything that inherits from torch.nn.module
#     """
#     dummy = torch.tensor(
#         np.zeros([2] + list(input_shape)),  # batchnorm requires batchsize >1
#         dtype=torch.float32,
#     )
#     out = dummy
#     module = [module] if not (isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList)) else module
#     for m in module:
#         # print(f'out {out.shape, m}')
#         out = m(out)
#         if isinstance(out, tuple):
#             out = out[0]
#     return out.shape[1:]


def get_output_shape(input_shape, model):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_shape)
        output = model(dummy_input)

    if was_training:
        model.train()

    return output.shape[1:]


class Localization2d(nn.Module):
    def __init__(self, input_shape, conv_channels):
        super().__init__()

        self.init_model(input_shape, conv_channels)
        out_shape = get_output_shape(input_shape, self.model)

        # создаем последний полносвязный слой
        expected_out_features = 6
        self.affine_param = nn.Linear(int(np.prod(out_shape)), expected_out_features)
        self.affine_param.weight.data.zero_()
        self.affine_param.bias.data.zero_()
        self.register_buffer("identity", torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

        if self.identity.numel() != self.affine_param.out_features:
            raise ValueError(
                f"identity has {self.identity.numel()} elements, "
                f"but affine_param outputs {self.affine_param.out_features}"
            )

    def init_model(self, input_shape, conv_channels):
        raise NotImplementedError

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.model(x)
        x = x.reshape(batch_size, -1)

        # зачем тут это нужно?
        delta_theta = self.affine_param(x)

        return delta_theta + self.identity
