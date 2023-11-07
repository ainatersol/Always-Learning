from torch import nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


## Vanilla CNN Classifier 

class ClassifierModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=2, padding=0), # N, C, H, W = Batch, C, H, W
            nn.ReLU(),
            nn.Conv2d(6, 8, kernel_size=1, stride=2, padding=0), # N, C, H, W = Batch,, C, H, W
            nn.ReLU(),
            nn.Flatten(), # N, L = = Batch, 32*32*32
            nn.Linear(int(input_dim**2/2) , 64),
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_enc = self.encoder(x)
        x_sig = self.sigmoid(x_enc)
        return x_enc, x_sig

class ClassifierModelXL(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.encoder = nn.Sequential(
                            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1),  # Keep the same spatial size
                            nn.BatchNorm2d(6),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2),  # Halve the spatial size

                            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2),  # Halve the spatial size
                            
                            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2),  # Halve the spatial size
                            nn.Flatten(), 

                            nn.Linear(in_features=int(input_dim**2/2), out_features=128),  
                            nn.ReLU(),
                            nn.Dropout(0.5),  # Add dropout for regularization

                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Dropout(0.5),  # Add dropout for regularization

                            nn.Linear(64, 16)
                                            )
        self.linear = nn.Linear(16, 1)

    def forward(self, x):
        x_enc = self.encoder(x)
        x_logits = self.linear(x_enc)
        return x_enc, x_logits


## CNN - ConvNeXt 
     
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
   
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x_z = self.forward_features(x)
        x = self.head(x_z)
        return x
            
class ConvNeXt_Z(ConvNeXt):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__(in_chans, num_classes, 
                 depths, dims, drop_path_rate, 
                 layer_scale_init_value, head_init_scale)

    def forward(self, x):
        x_z = self.forward_features(x)
        x = self.head(x_z)
        return x_z, x

def transfer_learning(pretrained_model, modified_model):
    pretrained_dict = pretrained_model.state_dict()
    modified_dict = modified_model.state_dict()

    # Transfer weights from pretrained model to new model where names match
    for name, param in pretrained_dict.items():
        if name in modified_dict and param.size() == modified_dict[name].size():
            modified_dict[name].copy_(param)
    modified_model.load_state_dict(modified_dict)
    return modified_model
    
def convnext_pretrained(z_dim=True, **kwargs):
    pretrained_model = ConvNeXt(in_chans=3, num_classes=21841, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
    checkpoint = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth", map_location="cpu")
    pretrained_model.load_state_dict(checkpoint["model"])
    if z_dim:
        modified_model = ConvNeXt_Z(**kwargs)
        return transfer_learning(pretrained_model, modified_model)
    else:
        return pretrained_model

