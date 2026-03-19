# Copyright 2026 pzihan. Licensed under the MIT License.

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal

class ResBlock(nn.Cell):
    """
    Basic Residual Block for 1D data
    """
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResBlock, self).__init__()
        self.stride = stride
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, pad_mode='pad', padding=padding, has_bias=False, weight_init=HeNormal())
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                               stride=1, pad_mode='pad', padding=padding, has_bias=False, weight_init=HeNormal())
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.SequentialCell([
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=False, weight_init=HeNormal()),
                nn.BatchNorm1d(out_channels)
            ])

    def construct(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

class SisFallResNet(nn.Cell):
    """
    1D ResNet for SisFall Dataset Activity Recognition
    Args:
        in_channels (int): Number of input sensor channels (Default 6 for SisFall: 3x Accel + 3x Gyro)
        num_classes (int): Number of activity classes (Default 34: 19 ADL + 15 Falls)
    """
    def __init__(self, in_channels=6, num_classes=34, base_filters=64):
        super(SisFallResNet, self).__init__()
        
        self.base_filters = base_filters
        
        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, 
                               pad_mode='pad', padding=3, has_bias=False, weight_init=HeNormal())
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, pad_mode='same')
        
        # ResNet Layers
        self.layer1 = self._make_layer(base_filters, base_filters, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_filters * 2, base_filters * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_filters * 4, base_filters * 4, blocks=2, stride=2)
        
        # Classification Head
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(base_filters * 4, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.SequentialCell(layers)

    def construct(self, x):
        # Input shape expectation: (N, C, L) or (N, L, C)
        # MindSpore Conv1d expects (N, C_in, L_in)
        # If input is (N, L, C), we define transpose here to be safe
        if x.ndim == 3 and x.shape[2] == 9: # Assuming C is last dimension if it's 9
             x = ops.transpose(x, (0, 2, 1))
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x

if __name__ == "__main__":
    import numpy as np
    from mindspore import Tensor
    
    # Test the model with dummy data
    # Batch size: 32, Channels: 6, Sequence Length: 256 (arbitrary window size)
    net = SisFallResNet(in_channels=6, num_classes=34)
    input_data = Tensor(np.random.randn(32, 6, 256).astype(np.float32))
    output = net(input_data)
    
    print("Network Architecture for SisFall:")
    # print(net) 
    print(f"\nInput shape: {input_data.shape}")
    print(f"Output shape: {output.shape}") 
    print("Model construction successful.")
