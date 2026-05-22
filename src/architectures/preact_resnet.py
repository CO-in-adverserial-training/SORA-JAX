import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple, Optional, Callable


class PreActBlock(nn.Module):
    """Pre‑activation basic residual block (two 3×3 convolutions)."""
    planes: int          # output channels
    stride: int = 1      # stride for the first convolution
    expansion: int = 1   # expansion factor (1 for BasicBlock)

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        in_planes = x.shape[-1]          # assuming NHWC format

        # pre‑activation for the residual path
        out = nn.BatchNorm(use_running_average=not training)(x)
        out = nn.relu(out)

        # first convolution (may downsample)
        out = nn.Conv(self.planes, kernel_size=(3, 3),
                      strides=(self.stride, self.stride),
                      padding='SAME', use_bias=False)(out)

        # second convolution (pre‑activation before it)
        out = nn.BatchNorm(use_running_average=not training)(out)
        out = nn.relu(out)
        out = nn.Conv(self.planes * self.expansion, kernel_size=(3, 3),
                      strides=(1, 1), padding='SAME', use_bias=False)(out)

        # shortcut (identity or projection)
        if self.stride != 1 or in_planes != self.planes * self.expansion:
            shortcut = nn.Conv(self.planes * self.expansion, kernel_size=(1, 1),
                               strides=(self.stride, self.stride),
                               use_bias=False)(x)
        else:
            shortcut = x

        return out + shortcut


class PreActBottleneck(nn.Module):
    """Pre‑activation bottleneck block (1×1 → 3×3 → 1×1)."""
    planes: int          # middle channels (output = planes * expansion)
    stride: int = 1      # stride for the 3×3 convolution
    expansion: int = 4   # expansion factor for the output channels

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        in_planes = x.shape[-1]
        out_planes = self.planes * self.expansion

        # shortcut (applied to the input x, before any pre‑activation)
        if self.stride != 1 or in_planes != out_planes:
            shortcut = nn.Conv(out_planes, kernel_size=(1, 1),
                               strides=(self.stride, self.stride),
                               use_bias=False)(x)
        else:
            shortcut = x

        # pre‑activation for the first 1×1 conv
        out = nn.BatchNorm(use_running_average=not training)(x)
        out = nn.relu(out)
        out = nn.Conv(self.planes, kernel_size=(1, 1), use_bias=False)(out)

        # pre‑activation for the 3×3 conv
        out = nn.BatchNorm(use_running_average=not training)(out)
        out = nn.relu(out)
        out = nn.Conv(self.planes, kernel_size=(3, 3),
                      strides=(self.stride, self.stride),
                      padding='SAME', use_bias=False)(out)

        # pre‑activation for the last 1×1 conv
        out = nn.BatchNorm(use_running_average=not training)(out)
        out = nn.relu(out)
        out = nn.Conv(out_planes, kernel_size=(1, 1), use_bias=False)(out)

        return out + shortcut


class PreActResNet(nn.Module):
    """Full Pre‑Activation ResNet."""
    block: Callable          # PreActBlock or PreActBottleneck
    num_blocks: Sequence[int]  # number of blocks in each layer
    num_classes: int = 10
    in_channels: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Initial convolution (no pre‑activation before it)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1),
                    padding='SAME', use_bias=False)(x)

        # Build the four main layers
        x = self._make_layer(self.block, 64, self.num_blocks[0],
                             stride=1, training=training)(x)
        x = self._make_layer(self.block, 128, self.num_blocks[1],
                             stride=2, training=training)(x)
        x = self._make_layer(self.block, 256, self.num_blocks[2],
                             stride=2, training=training)(x)
        x = self._make_layer(self.block, 512, self.num_blocks[3],
                             stride=2, training=training)(x)

        # Final pre‑activation, global average pooling and classification
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))   # global average pool (NHWC → NC)
        x = nn.Dense(self.num_classes)(x)
        return x

    def _make_layer(self, block: Callable, planes: int, num_blocks: int,
                    stride: int, training: bool):
        """Create a layer consisting of several residual blocks."""
        layers = []
        # First block may downsample
        layers.append(block(planes, stride=stride))
        # Remaining blocks have stride 1 and the same number of planes
        for _ in range(1, num_blocks):
            layers.append(block(planes, stride=1))

        class Sequential(nn.Module):
            blocks: Sequence[nn.Module]

            @nn.compact
            def __call__(self, x, training):
                for block in self.blocks:
                    x = block(x, training=training)
                return x

        return Sequential(blocks=layers)


# Convenience constructors for different PreActResNet variants
def PreActResNet18(**kwargs) -> PreActResNet:
    return PreActResNet(PreActBlock, [2, 2, 2, 2], **kwargs)

def PreActResNet34(**kwargs) -> PreActResNet:
    return PreActResNet(PreActBlock, [3, 4, 6, 3], **kwargs)

def PreActResNet50(**kwargs) -> PreActResNet:
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], **kwargs)

def PreActResNet101(**kwargs) -> PreActResNet:
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], **kwargs)

def PreActResNet152(**kwargs) -> PreActResNet:
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], **kwargs)
