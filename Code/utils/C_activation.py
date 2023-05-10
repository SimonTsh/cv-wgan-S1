import torch
import torch.nn as nn
import torch.nn.functional as F

class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_i = nn.LeakyReLU()
        self.relu_r = nn.LeakyReLU()

    def forward(self, z):
        return self.relu_r(z.real) + self.relu_i(z.imag) * 1j

class CPReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu_r = nn.PReLU()
        self.prelu_i = nn.PReLU()

    def forward(self, z):
        return self.prelu_r(z.real) + 1j*self.prelu_i(z.imag)

class CELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(CELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        output_r = F.elu(x.real, self.alpha)
        output_i = F.elu(x.imag, self.alpha)
        return torch.complex(output_r, output_i)

class CGELU(nn.Module):
    def __init__(self):
        super(CGELU, self).__init__()

    def forward(self, x):
        output_r = F.gelu(x.real)
        output_i = F.gelu(x.imag)
        return torch.complex(output_r, output_i)

class zReLU(nn.Module):
    def forward(self, z):
        pos_real = z.real > 0
        pos_img = z.imag > 0
        return z * pos_real * pos_img
    
class zLeakyReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.parameter.Parameter(data=torch.Tensor([0.2]), requires_grad = True)

    def forward(self, z):
        pos_real = z.real > 0
        pos_img = z.imag > 0
        return z * pos_real * pos_img + self.a*(z * ~(pos_real * pos_img))

class zAbsReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.radius = torch.nn.parameter.Parameter(data=torch.Tensor([1.]), requires_grad = True)

    def forward(self, z):
        mask = z.abs() < self.radius
        return z * mask

class zAbsLeakyReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.radius = torch.nn.parameter.Parameter(data=torch.Tensor([1.]), requires_grad = True)
        self.a = torch.nn.parameter.Parameter(data=torch.Tensor([0.]), requires_grad = True)
    
    def forward(self, z):
        mask = z.abs() < self.radius
        return z * mask + self.a*(z * ~(mask))

class zAngleLeakyReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.angle = torch.nn.parameter.Parameter(data=torch.Tensor(torch.distributions.uniform.Uniform(-torch.pi, torch.pi).sample([1,])), requires_grad = True)
        self.width = torch.nn.parameter.Parameter(data=torch.Tensor([torch.pi/2]), requires_grad = True)
        self.a = torch.nn.parameter.Parameter(data=torch.Tensor([-0.04]), requires_grad = True)

    def forward(self, z):
        mask_1 = z.angle() < (self.angle + self.width/2)
        mask_2 = z.angle() > (self.angle - self.width/2)
        mask = mask_1*mask_2
        return z * mask + self.a*(z * ~(mask))

class zAbsAngleLeakyReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.angle = zAngleLeakyReLU()
        self.abs = zAbsLeakyReLU()

    def forward(self, z):
        return self.abs(self.angle(z))

class Mod(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return torch.abs(z)

class modReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.b = torch.nn.Parameter(torch.tensor(0., dtype=torch.float), True)
    
    def forward(self, z):
        return nn.functional.relu(z.abs() + self.b)*torch.exp(1j*z.angle())

class modLeakyReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.b = torch.nn.Parameter(torch.tensor(-1., dtype=torch.float), True)
    
    def forward(self, z):
        return nn.functional.leaky_relu(z.abs() + self.b, 0.05)*torch.exp(1j*z.angle())

class C_RealSigmoid(nn.Module):
    '''
    Sigmoid on real part.
    '''
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z):
        return torch.sigmoid(z.real) + 1j*z.imag

class C_ImagSigmoid(nn.Module):
    '''
    Sigmoid on real part.
    '''
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z):
        return z.real + 1j*torch.sigmoid(z.imag)

class Cardioid(nn.Module):
    '''
    Cardioid activation function.
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, z):
        return 1/2*(1+torch.cos(z.angle()))*z
    
class C_Tanh(nn.Module):
    '''
    Complex Tanh activation function.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, z):
        return self.tanh(z.real) + 1j*self.tanh(z.imag)

class modTanh(nn.Module):
    '''
    Complex Tanh activation function.
    '''
    def forward(self, z):
        return torch.tanh(z.abs()) * torch.exp(z.angle()*1j)
    
class C_Sigmoid(nn.Module):
    '''
    Complex Sigmoid activation function.
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, z):
        return torch.sigmoid(z.real) + 1j*torch.sigmoid(z.imag)
    

def get_activation_function(activation:str="zReLU"):
    '''
    Returns activation function from a string.

    Args :
        - activation (str)

    Available activation functions (complex valued):
        - CReLU
        - zReLU
        - modReLU
    '''
    try:
        f = eval(f"{activation}()")
    except:
        print(f"Activation function {activation} not found. Using zReLU.")
        f = zReLU()

    return f