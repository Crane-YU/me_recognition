import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(input_tensor, dim=-1, epsilon=1e-7):
    norm = torch.norm(input_tensor, p=2, dim=dim, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (epsilon + norm)
    return scale * input_tensor


class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """

    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps  # dim_caps = 8 in primary capsule layer
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        outputs = self.conv2d(x)  # [bs, 32 * 8, 6, 6]
        outputs = outputs.permute(0, 2, 3, 1).contiguous()  # I add
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        return squash(outputs)


class MECapsule(nn.Module):
    """
    The ME capsule layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons.
    MECapsule just expands the output of the neuron from scalar to vector.
    :param in_num_caps: number of capsules input to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param num_iter: number of iterations for the routing algorithm
    """

    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, num_iter=3):
        super(MECapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.num_iter = num_iter
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps),
                                   requires_grad=True)  # [10, 1152, 16, 8]

    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)  # [bs, 10, 1152, 16]
        u_hat_detached = u_hat.detach()

        b = torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps).cuda()  # [bs, 10, 1152]

        for i in range(self.num_iter):
            c = F.softmax(b, dim=1)  # [bs, 10, 1152]

            if i == self.routings - 1:
                outputs = squash(torch.sum(c[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                outputs = squash(torch.sum(c[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))  # [bs, 10, 1, 16]
                b = b + torch.sum(outputs * u_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)  # [bs, 10, 16]
