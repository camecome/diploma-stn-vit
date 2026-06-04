import torch.nn as nn

from .modeling import Localization2d


class ViTLocalization(Localization2d):
    def init_model(self, input_shape, conv_channels):
        layers = []

        in_channels = input_shape[0]
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )

            in_channels = out_channels

        self.model = nn.Sequential(*layers)
