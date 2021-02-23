from torch import nn
from torchvision.models.resnet import resnet18


class resnet_cifar(nn.Module):
    r"""CIFAR-variant of ResNet18."""
    def __init__(self):
        super().__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)

        # encoder
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        y = self.f(x)
        return y.view(y.size(0), -1).contiguous()

    @staticmethod
    def get_linear_classifier(input_dim=512, output_dim=10):
        r"""Return linear classification layer."""
        return nn.Linear(input_dim, output_dim)
