import numpy as np
import torch
import torch.nn as nn


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

        # предсказываем только 4 параметра delta A
        expected_out_features = 4
        self.affine_param = nn.Linear(int(np.prod(out_shape)), expected_out_features)

        self.affine_param.weight.data.zero_()
        self.affine_param.bias.data.zero_()

    def init_model(self, input_shape, conv_channels):
        raise NotImplementedError

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.model(x)
        # flatten output
        x = x.reshape(batch_size, -1)

        delta_a = self.affine_param(x)

        return delta_a
