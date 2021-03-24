from .common_head import *

__all__ = ['ea_resnet34', 'ea_resnet50', 'ea_resnet101', 'ea_resnet152']


class EAAugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 k, v, Nh, shape, att_downsample, alpha=1.0, beta=0.5, relative=True):
        super(EAAugmentedConv, self).__init__()
        self.dk = int(out_channels * k)
        self.dv = int(out_channels * v)
        self.Nh = Nh
        self.shape = shape

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.att_downsample = att_downsample
        self.alpha = alpha
        self.beta = beta
        self.relative = relative

        self.conv_out = nn.Conv2d(self.in_channels,
                                  self.out_channels - self.dv,
                                  self.kernel_size,
                                  stride=stride,
                                  padding=1)

        if stride == 2:
            self.pool_input = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        if att_downsample:
            self.pool_att = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.upsample_att = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1)

        self.att_conv = nn.Sequential(
                nn.Conv2d(Nh, Nh, 3, stride=1, padding=1),
                nn.ReLU(inplace=True))

        self.attn_out = nn.Conv2d(self.dv, self.dv, 1)

        if self.relative:
            self.key_rel_w = nn.Parameter(
                    torch.randn((2 * self.shape - 1, self.dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(
                    torch.randn((2 * self.shape - 1, self.dk // Nh), requires_grad=True))

    def forward(self, x):
        assert isinstance(x, tuple) and len(x) == 2
        x, prev_att = x[0], x[1]
        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)

        if self.stride == 2:
            x = self.pool_input(x)

        if self.att_downsample:
            x = self.pool_att(x)

        # x for self-attention
        # (batch_size, channels, height, width)
        batch, _, height, width = x.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        if prev_att is not None:
            if prev_att.shape[1] == logits.shape[1] \
                    and prev_att.shape[2] == logits.shape[2] \
                    and prev_att.shape[3] == logits.shape[3]:
                att_matrix = (1 - self.beta) * logits + self.beta * prev_att
                logits = self.att_conv(att_matrix)
                logits = self.alpha * logits + (1 - self.alpha) * att_matrix

        # N C H W
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)

        if self.att_downsample:
            attn_out = self.upsample_att(attn_out)

        return (torch.cat((conv_out, attn_out), dim=1), logits)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        N, _, H, W = x.size()
        qkv = self.qkv_conv(x)
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 shape=None, att_downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class EABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 shape=None, att_downsample=None):
        super(EABasicBlock, self).__init__()
        self.conv1 = EAAugmentedConv(in_channels=inplanes, out_channels=planes,
                                     kernel_size=3, shape=shape, stride=stride,
                                     k=0.25, v=0.25, Nh=8,
                                     att_downsample=att_downsample)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = EAAugmentedConv(in_channels=planes, out_channels=planes,
                                     kernel_size=3, shape=shape, stride=1,
                                     k=0.25, v=0.25, Nh=8,
                                     att_downsample=att_downsample)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        assert isinstance(x, tuple) and len(x) == 2
        x, att = x[0], x[1]

        identity = x

        (out, att) = self.conv1((x, att))
        out = self.bn1(out)
        out = self.relu(out)

        (out, att) = self.conv2((out, att))
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return (out, att)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 shape=None, att_downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 shape=None, att_downsample=None):
        super(EABottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = EAAugmentedConv(in_channels=planes, out_channels=planes,
                                     kernel_size=3, shape=shape, stride=stride,
                                     k=0.25, v=0.125, Nh=8,
                                     att_downsample=att_downsample)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        assert isinstance(x, tuple) and len(x) == 2
        x, att = x[0], x[1]

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        (out, att) = self.conv2((out, att))
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return (out, att)


class EAResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(EAResNet, self).__init__()
        self.shape = 224 # original shape is 224
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.shape = self.shape // 4
        self.layer1 = self._make_layer(block, 64, layers[0], layer_idx=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, layer_idx=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, layer_idx=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, layer_idx=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, EABottleneck) or isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, EABasicBlock) or isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, layer_idx, stride=1):
        shape = None
        att_downsample = None
        if layer_idx >= 1:
            if layer_idx == 1:
                att_downsample = True
                shape = self.shape // 4
            else:
                att_downsample = False
                shape = self.shape // 2
            self.shape = self.shape // 2
            if block is BasicBlock:
                block = EABasicBlock
            elif block is Bottleneck:
                block = EABottleneck

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            shape=shape, att_downsample=att_downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, shape=shape,
                                att_downsample=att_downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x, att = self.layer2((x, None))
        x, att = self.layer3((x, att))
        x, att = self.layer4((x, att))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ea_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = EAResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def ea_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = EAResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def ea_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = EAResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def ea_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = EAResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
