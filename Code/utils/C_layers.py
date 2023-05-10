import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt

try:
    from utils.C_activation import *
except:
    from C_activation import *

class C_Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, z):
        mask = torch.nn.functional.dropout(
            torch.ones(z.shape), self.p, training=self.training
        ).to(z.device)
        return mask * z

class C_Dropout2d(nn.Module):
    def __init__(self, p=0.5, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.p = p

    def forward(self, z):
        mask = torch.nn.functional.dropout2d(
            torch.ones(z.shape), self.p, training=self.training
        ).to(self.device)
        return mask * z

class C_MaxPool2d(nn.Module):
    '''
    Implementation of torch.nn.MaxPool2d for complex numbers.
    Apply MaxPool2d on the module of the input.
    Returns complex values associated to the MaxPool indices.
    '''
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) -> None:
        super().__init__()
        self.return_indices = return_indices
        self.m = torch.nn.MaxPool2d(kernel_size,
                                    stride,
                                    padding,
                                    dilation,
                                    ceil_mode=ceil_mode,
                                    return_indices=True)


    def forward(self, z: torch.Tensor):
        _, indices = self.m(torch.abs(z))
        
        if self.return_indices:
            return z.flatten()[indices], indices
        else :
            return z.flatten()[indices]

class C_AvgPool2d(nn.Module):
    '''
    Implementation of torch.nn.AvgPool2d for complex numbers.
    Apply AvgPool2d on the real and imaginary part.
    Returns complex values associated to the AvgPool2dresults.
    '''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> None:
        super().__init__()
        if type(kernel_size) == int:
            self.kernel_size = [kernel_size]*2 + [1]
        elif type(kernel_size) == tuple:
            if len(kernel_size) < 3:
                self.kernel_size = [kernel_size] + [1]
            else :
                self.kernel_size = kernel_size

        if type(stride) == int:
            self.stride = [stride]*2 + [1]
        elif type(stride) == tuple:
            if len(stride) < 3:
                self.stride = [stride] + [1]
            else :
                self.stride = stride

        self.m = torch.nn.AvgPool3d(self.kernel_size,
                                    self.stride,
                                    padding,
                                    ceil_mode,
                                    count_include_pad,
                                    divisor_override)

    def forward(self, z: torch.Tensor):
        return torch.view_as_complex(self.m(torch.view_as_real(z)))

class C_ConvTranspose2d(nn.Module):
    '''
    Implementation of torch.nn.Conv2dTranspose for complex numbers.
    Apply Conv2dTranspose on real and imaginary part of the complex number.
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__()

        self.m_real = torch.nn.ConvTranspose2d( in_channels,
                                                out_channels,
                                                kernel_size,
                                                stride,
                                                padding,
                                                output_padding,
                                                groups,
                                                bias,
                                                dilation,
                                                padding_mode,
                                                device,
                                                dtype)

        self.m_imag = torch.nn.ConvTranspose2d( in_channels,
                                                out_channels,
                                                kernel_size,
                                                stride,
                                                padding,
                                                output_padding,
                                                groups,
                                                bias,
                                                dilation,
                                                padding_mode,
                                                device,
                                                dtype)

    def forward(self, z: torch.Tensor):
        return torch.view_as_complex(torch.cat((torch.unsqueeze(self.m_real(z.real) - self.m_imag(z.imag), -1) ,
                                                        torch.unsqueeze(self.m_real(z.imag) + self.m_imag(z.real), -1)),
                                                        axis=-1)
                                    )

class C_UpsampleConv2d(nn.Module):
    '''
    Upsample complex value image and apply a 2d convolution.
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        upsampling_mode: str = 'nearest',
        scale_factor = (2, 2),
        device=None,
        dtype=None) -> None:

        super().__init__()
        self.m = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    padding,
                                    dilation,
                                    groups,
                                    bias,
                                    padding_mode,
                                    device,
                                    dtype)
        
        self.upsampling_mode = upsampling_mode
        self.scale_factor = scale_factor
        self.up = lambda x : torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.upsampling_mode)

    def forward(self, z: torch.Tensor):
        upsampled = torch.view_as_complex(torch.cat((torch.unsqueeze(self.up(z.real), -1) ,
                                                             torch.unsqueeze(self.up(z.imag), -1)),
                                                             axis=-1)
                                        )
        return self.m(upsampled)


class C_BatchNorm(nn.Module):
    '''
    BatchNorm for complex valued neural networks.
    Input of shape [B, C, H, W]
    '''
    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        device=None,
        ) -> None:
        super().__init__()

        self.affine = affine
        self.device = device
        self.momentum = momentum
        self.eps = eps

        self.running_avg_initilized = False

        I = torch.eye(2)/torch.sqrt(torch.tensor(2))
        self.gamma = torch.nn.Parameter(I, requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros(2), requires_grad=True)

        self.gamma.to(device)
        self.beta.to(device)

    def init_running_avg(self, n_features):

        I = torch.eye(2)/torch.sqrt(torch.tensor(2))
        # [C, H, W, 2, 2]
        self.sigma_running = self.sigma = I.tile(n_features, 1, 1)
        # [C, H, W, 2]
        self.mu_running = self.mu = torch.zeros(1,2).tile(n_features, 1)

    def update_mu_sigma(self, z:torch.Tensor):
        '''
        Update mu and sigma matrix.
        '''
        # [B, C, H, W, 2]
        B, C, H, W, _ = z.shape
        # [B, C*H*W, 2]
        z = z.reshape(B, C*H*W, 2)

        ### MEAN
        # [1, C*H*W, 2]
        self.mu = torch.mean(z, dim=0).unsqueeze(0)

        ### COV
        # [B, C*H*W, 2]
        x_centered = z - self.mu
        # [B*C*H*W, 2]
        x_centered_b = x_centered.reshape(B*C*H*W, 2)
        # [B*C*H*W, 2] = [B*C*H*W, 2, 1] @ [B*C*H*W, 1, 2]
        prods = torch.bmm(x_centered_b.unsqueeze(2), x_centered_b.unsqueeze(1))
        # [B, C*H*W, 2, 2]
        prods = prods.view(B, C*H*W, 2, 2)
        # [1, C*H*W, 2, 2]
        self.sigma = 1/(B-1)*prods.sum(dim=0)

        self.sigma_running = (1-self.momentum) * self.sigma_running + self.momentum * self.sigma
        self.mu_running = (1-self.momentum) * self.mu_running + self.momentum * self.mu

    def inv_sqrt_22_batch(self, M:torch.Tensor):
        '''
        Return the square root of the inverse of a matrix of shape [N, 2, 2]
        '''
        N = M.shape[0]
        # [M, 1, 1]
        trace = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1).reshape(N, 1, 1)
        det = torch.linalg.det(M).reshape(N, 1, 1)

        s = torch.sqrt(det)
        t = torch.sqrt(trace + 2*s)
        # [M, 2, 2]
        M_inv = 1/t * (M + s*torch.eye(2).tile(N, 1, 1))
        return M_inv

    def normalize(self, z:torch.Tensor):
        '''
        Returns a normalized tensor of input z (complex) [B, C, H, W].
        Uses self.sigma as covariance matrix and self.mu as mean vector.
        Return:
            torch.Tensor [B*C*H*W, 2] (complex viewed as real)
        '''
        B, C, H, W, _ = z.shape
        z = z.reshape(B, C*H*W, 2)
        # [C*H*W, 2, 2]
        sigma_inv_sqrt = self.inv_sqrt_22_batch(self.sigma)
        # [B, C*H*W, 2, 1]
        z_centered = (z - self.mu).reshape(B*C*H*W, 2).unsqueeze(-1)
        o = torch.bmm(sigma_inv_sqrt.tile(B, 1, 1).reshape(B*C*H*W, 2, 2), z_centered)
        return o

    def normalize_inference(self, z:torch.Tensor):
        '''
        Returns a normalized tensor of input z (complex) [B, C, H, W].
        Uses self.sigma_running as covariance matrix and self.mu_running as mean vector.
        Return:
            torch.Tensor [B*C*H*W, 2] (complex viewed as real)
        '''
        B, C, H, W, _ = z.shape
        # [C*H*W, 2, 2]
        sigma_inv_sqrt = self.inv_sqrt_22_batch(self.sigma_running)
        # [B, C*H*W, 2, 1]
        z_centered = (z - self.mu_running).reshape(B*C*H*W, 2).unsqueeze(-1)
        o = torch.bmm(sigma_inv_sqrt.tile(B, 1, 1).reshape(B*C*H*W, 2, 2), z_centered)
        return o

    def forward(self, z : torch.Tensor):
        # z : [B, C, H, W] (complex)

        B, C, H, W = z.shape
        if not(self.running_avg_initilized):
            self.init_running_avg(C*H*W)

        z = torch.view_as_real(z)

        if self.training:
            self.update_mu_sigma(z)
            # [B*C*H*W, 2]
            o = self.normalize(z)
        else :
            # [B*C*H*W, 2]
            o = self.normalize_inference(z)
        
        if self.affine:
            # [B*C*H*W, 2, 1]
            o = torch.bmm(self.gamma.tile(B*C*H*W, 1, 1), o) + self.beta.reshape(1, 2, 1)

        # [B, C, H, W, 2]
        o = o.view(B, C, H, W, 2)
        return torch.view_as_complex(o)

class C_BatchNorm2d(nn.Module):
    '''
    BatchNorm for complex valued neural networks.
    Input of shape [B, C, H, W]
    '''
    def __init__(
        self,
        num_features,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        ) -> None:
        super().__init__()

        self.num_features = num_features
        self.affine = affine
        #self.momentum = momentum
        self.register_buffer("momentum", torch.tensor(momentum))
        self.eps = eps


        self.register_buffer("id2", torch.eye(2))
        # [C, 2, 2]
        self.register_buffer("sigma_running", torch.eye(2)/torch.sqrt(torch.tensor(2)).tile(num_features, 1, 1))
        self.register_buffer("sigma", torch.eye(2)/torch.sqrt(torch.tensor(2)).tile(num_features, 1, 1))
        # [C, 2]
        self.register_buffer("mu_running", torch.zeros(1,2).tile(num_features, 1).reshape(1, num_features, 1, 1, 2))
        self.register_buffer("mu", torch.zeros(1,2).tile(num_features, 1).reshape(1, num_features, 1, 1, 2))
        #self.mu_running = self.mu = torch.zeros(1,2).tile(num_features, 1)

        self.gamma = torch.nn.Parameter(torch.eye(2)/torch.sqrt(torch.tensor(2)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros(2), requires_grad=True)

    def update_mu_sigma(self, z:torch.Tensor):
        '''
        Update mu and sigma matrix.
        '''
        B, C, H, W, _ = z.shape

        ### MEAN
        # [1, C, 1, 1, 2]
        self.mu = torch.mean(z, dim=(0, 2, 3), keepdim=True)

        ### COV
        # [B, C, H, 2]
        x_centered = z - self.mu
        # [B*C*H*W, 2]
        x_centered_b = x_centered.reshape(B*C*H*W, 2)

        # [B*C*H*W, 2, 2] = [B*C*H*W, 2, 1] @ [B*C*H*W, 1, 2]
        prods = torch.bmm(x_centered_b.unsqueeze(2), x_centered_b.unsqueeze(1))
        prods = prods.view(B, C, H, W, 2, 2)
        # [C, 2, 2]
        self.sigma = 1/(B*H*W)*prods.sum(dim=(0, 2, 3)).squeeze() + self.id2.tile(C, 1, 1)*self.eps

        self.sigma_running = (1-self.momentum) * self.sigma_running + self.momentum * self.sigma
        self.mu_running = (1-self.momentum) * self.mu_running + self.momentum * self.mu

    def inv_sqrt_22_batch(self, M:torch.Tensor):
        '''
        Return the square root of the inverse of a matrix of shape [N, 2, 2]
        '''
        N = M.shape[0]
        # [M, 1, 1]
        trace = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1).reshape(N, 1, 1)
        #det = torch.linalg.det(M).reshape(N, 1, 1)
        det = (M[:, 0, 0]*M[:, 1, 1] - M[:, 0, 1]*M[:, 1, 0]).reshape(N, 1, 1)

        s = torch.sqrt(det)
        t = torch.sqrt(trace + 2*s)
        # [M, 2, 2]
        M_sqrt = 1/t * (M + s*self.id2.tile(N, 1, 1))
        M_inv = torch.linalg.inv(M_sqrt)
        return M_inv

    def normalize(self, z:torch.Tensor):
        '''
        Returns a normalized tensor of input z (complex) [B, C, H, W].
        Uses self.sigma as covariance matrix and self.mu as mean vector.
        Return:
            torch.Tensor [B*C*H*W, 2] (complex viewed as real)
        '''
        B, C, H, W, _ = z.shape
        # [C, 2, 2]
        sigma_inv_sqrt = self.inv_sqrt_22_batch(self.sigma)
        # [B, C, H, W, 2, 1]
        z_centered = (z - self.mu).reshape(B*C*H*W, 2).unsqueeze(-1)
        o = torch.bmm(sigma_inv_sqrt.tile(B*H*W, 1, 1).reshape(B*C*H*W, 2, 2), z_centered)
        return o

    def normalize_inference(self, z:torch.Tensor):
        '''
        Returns a normalized tensor of input z (complex) [B, C, H, W].
        Uses self.sigma_running as covariance matrix and self.mu_running as mean vector.
        Return:
            torch.Tensor [B*C*H*W, 2] (complex viewed as real)
        '''
        B, C, H, W, _ = z.shape
        # [C, 2, 2]
        sigma_inv_sqrt = self.inv_sqrt_22_batch(self.sigma_running)
        # [B, C, H, W, 2, 1]
        z_centered = (z - self.mu_running.reshape(1, C, 1, 1, 2)).reshape(B*C*H*W, 2).unsqueeze(-1)
        o = torch.bmm(sigma_inv_sqrt.tile(B*H*W, 1, 1).reshape(B*C*H*W, 2, 2), z_centered)
        return o

    def forward(self, z : torch.Tensor):
        # z : [B, C, H, W] (complex)

        B, C, H, W = z.shape

        z = torch.view_as_real(z)

        if self.training:
            self.update_mu_sigma(z)
            # [B*C*H*W, 2]
            o = self.normalize(z)
        else :
            # [B*C*H*W, 2]
            o = self.normalize_inference(z)
        
        if self.affine:
            # [B*C*H*W, 2, 1]
            o = torch.bmm(self.gamma.tile(B*C*H*W, 1, 1), o) + self.beta.reshape(1, 2, 1)

        # [B, C, H, W, 2]
        o = o.view(B, C, H, W, 2)
        return torch.view_as_complex(o)
    
def get_convolution_block(in_channels:int=1,
                          out_channels:int=1,
                          kernel_size_conv:int=3,
                          stride_conv:int=1,
                          padding_conv:int=1,
                          kernel_size_pool:int=2,
                          stride_pool:int=2,
                          padding_pool:int=0,
                          batch_norm:bool=True,
                          activation:str="CReLU",
                          dtype=torch.complex64
                          ):
    '''
    Construct a convolutional block with the following layers :
    IN -> Conv2d -> activation -> C_BatchNorm -> (Pooling) -> OUT
    '''
    layers = []

    ### Conv2d
    conv_layer = nn.Conv2d( in_channels,
                            out_channels,
                            kernel_size_conv,
                            stride_conv,
                            padding_conv,
                            dtype=dtype)
    layers.append(conv_layer)

    ### Activation
    if activation.lower() != "none":
        activation_f = get_activation_function(activation)
        layers.append(activation_f)

    ### C_BatchNorm
    if batch_norm:
        layers.append(C_BatchNorm2d(out_channels))
    
    ### Pooling
    if kernel_size_pool != 0:
        pool_layer = C_MaxPool2d(kernel_size_pool,
                                stride_pool,
                                padding_pool)
        layers.append(pool_layer)
        
    m = nn.Sequential(*layers)

    return m

def cov(x, y):
    # [N]
    N = x.shape[0]
    return 1/(N-1)*(x-torch.mean(x, dim=0))@(y-torch.mean(y, dim=0)).T

def test_BatchNorm2d():

    B = 2
    C = 2
    H = 20
    W = 20

    bn = C_BatchNorm2d(C, affine=False)

    A = torch.rand(B, C, H, W, 1,)
    B = A/2 + torch.rand(B, C, H, W, 1,)
    input = torch.view_as_complex(torch.cat((A, B), dim=-1))

    for i in range(1000):
        o = bn(input)
        if o.isnan().any():
            print("nan")
    bn.eval()
    o = bn(input)
    o = torch.view_as_real(o)

    X = o[:, 0, :, :, 0].flatten()
    Y = o[:, 0, :, :, 0].flatten()
    c = cov(X, Y)
    print("cov(Re(X[:,0,:,:], Re(X[:,0,:,:]):", c.item())

    X = o[:, 0, :, :, 0].flatten()
    Y = o[:, 0, :, :, 1].flatten()
    c = cov(X, Y)
    print("cov(Re(X[:,0,:,:], Im(X[:,0,:,:]):", c.item())

    X = o[:, 0, :, :, 1].flatten()
    Y = o[:, 0, :, :, 1].flatten()
    c = cov(X, Y)
    print("cov(Im(X[:,0,:,:], Im(X[:,0,:,:]):", c.item())


    X = o[:, 1, :, :, 0].flatten()
    Y = o[:, 0, :, :, 0].flatten()
    c = cov(X, Y)
    print("cov(Re(X[:,1,:,:], Re(X[:,0,:,:]):", c.item())

    print(bn.sigma_running)
    print(bn.mu_running)

    print(bn.gamma)
    print(bn.beta)

    # Plot the points in blue and red
    x1, y1 = input.real.squeeze(), input.imag.squeeze()
    x2, y2 = o[:,:,:,:,0].squeeze(), o[:,:,:,:,1].squeeze()

    fig, ax = plt.subplots()
    ax.scatter(x1, y1, c='blue', label='input', marker=".")
    ax.scatter(x2, y2, c='red', label='normalized', marker=".")
    ax.legend()
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('Complex Batch Normalization')
    plt.savefig("./logs/_figures/batch_norm.png")

if __name__ == "__main__":

    test_BatchNorm2d()