from .softmax_transformer import SoftmaxTransformer
from .kernel_transformer import KernelTransformer
from .ens_transformer import EnsTransformer
from .conv import EarthPadding, EnsConv2d


avail_transformers = {
    'softmax': SoftmaxTransformer,
    'kernel': KernelTransformer,
    'ensemble': EnsTransformer
}
