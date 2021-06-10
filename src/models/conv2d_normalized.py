from torch import norm
from torch.nn import Conv2d


class Conv2d_l2_normalized(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, l2_norm_constant=1., **kwargs):
        super(Conv2d_l2_normalized, self).__init__(
            in_channels, out_channels, kernel_size, **kwargs)

        self.weight.data /= (norm(self.weight, p=2,
                                  dim=(1, 2, 3), keepdim=True)/l2_norm_constant)
