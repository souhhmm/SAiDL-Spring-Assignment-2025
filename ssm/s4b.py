import torch
from torch import nn
from torch import view_as_real
from torch.fft import ifft, irfft, rfft
from torch.nn import functional as F


def log_step_init(tensor, dt_min=0.001, dt_max=0.1):
    scale = torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min))
    return tensor * scale + torch.log(torch.tensor(dt_min))

def hippo(N):
    P = torch.sqrt(1 + 2 * torch.arange(1, N+1, dtype=torch.float))
    A = torch.outer(P, P)
    A = torch.tril(A) - torch.diag(torch.arange(1, N+1, dtype=torch.float))
    return A

def hippo_dplr(N):
    A = -1 * hippo(N)  # -ve sign here

    p = 0.5 * torch.sqrt(2 * torch.arange(1, N+1, dtype=torch.float32) + 1.0)
    p = p.to(torch.complex64)
    
    Ap = A.to(torch.complex64) + torch.outer(p, p)
    
    # eigen values, vectors
    lambda_, V = torch.linalg.eig(Ap)

    return lambda_, p, V


def p_lambda(n):
    lambda_, p, V = hippo_dplr(n)
    Vc = V.conj().T
    p = Vc @ p
    return [p, lambda_]


def cauchy_kernel(v, omega, lambda_):
    if v.ndim == 1:
        v = v.unsqueeze(0).unsqueeze(0)
    elif v.ndim == 2:
        v = v.unsqueeze(1)
    return (v/(omega-lambda_)).sum(dim=-1)


def causal_convolution(u, K):
    l_max = u.shape[1]  # u.shape = [batch, seq_length, d_model]
    
    # pad seq_length with l_max zeroes and compute fft
    ud = rfft(F.pad(u.float(), pad=(0, 0, 0, l_max, 0, 0)), dim=1)
    Kd = rfft(F.pad(K.float(), pad=(0, l_max)), dim=-1)
    
    # freq -> time domain
    return irfft(ud.transpose(-2, -1)*Kd)[..., :l_max].transpose(-2, -1).type_as(u)

# compute frequencies
def f_omega(l_max, dtype=torch.complex64):
    return torch.arange(l_max).type(dtype).mul(2j * torch.tensor(torch.pi) / l_max).exp()


class S4Layer(nn.Module):
    def __init__(self, d_model, n, l_max):
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.l_max = l_max

        p, lambda_ = p_lambda(n)
        p = p.to(torch.complex64)
        lambda_ = lambda_.to(torch.complex64)
        self._p = nn.Parameter(view_as_real(p))
        self._lambda_ = nn.Parameter(view_as_real(lambda_).unsqueeze(0).unsqueeze(1))
        
        # make non trainable
        self.register_buffer(
            "omega",
            tensor=f_omega(self.l_max, dtype=torch.complex64),
        )
        self.register_buffer(
            "ifft_order",
            tensor=torch.tensor(
                [i if i == 0 else self.l_max-i for i in range(self.l_max)],
                dtype=torch.long,
            ),
        )
        
        B_init = torch.sqrt(2 * torch.arange(1, n+1, dtype=torch.float32) + 1.0)
        B_init = B_init.repeat(d_model, 1)
        self._B = nn.Parameter(
            view_as_real(B_init.to(torch.complex64))
        )
        self._Ct = nn.Parameter(
            view_as_real(
                nn.init.xavier_normal_(torch.empty(d_model, n, dtype=torch.complex64))
            )
        )
        self.D = nn.Parameter(torch.ones(1, 1, d_model))
        self.log_step = nn.Parameter(log_step_init(torch.rand(d_model)))

    @property
    def p(self):
        return torch.view_as_complex(self._p)

    @property
    def lambda_(self):
        return torch.view_as_complex(self._lambda_)

    @property
    def B(self):
        return torch.view_as_complex(self._B)

    @property
    def Ct(self):
        return torch.view_as_complex(self._Ct)

    # algorithm 1
    def roots(self):
        a0 = self.Ct.conj()
        a1 = self.p.conj()
        b0 = self.B
        b1 = self.p
        step = self.log_step.exp()

        # bilinear discretization
        g = torch.outer(2.0/step, (1.0-self.omega)/(1.0+self.omega))
        c = 2.0/(1.0+self.omega)

        k00 = cauchy_kernel(a0*b0, g.unsqueeze(-1), self.lambda_)
        k01 = cauchy_kernel(a0*b1, g.unsqueeze(-1), self.lambda_)
        k10 = cauchy_kernel(a1*b0, g.unsqueeze(-1), self.lambda_)
        k11 = cauchy_kernel(a1*b1, g.unsqueeze(-1), self.lambda_)
        return c*(k00-k01*(1.0/(1.0+k11))*k10)

    @property
    def K(self):
        at_roots = self.roots()
        out = ifft(at_roots, n=self.l_max, dim=-1)
        conv = torch.stack([i[self.ifft_order] for i in out]).real
        return conv.unsqueeze(0)

    def forward(self, u):
        return causal_convolution(u, K=self.K) + (self.D * u)


class S4Block(nn.Module):
    def __init__(
        self,
        d_model,
        n,
        l_max,
        dropout=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.s4 = S4Layer(d_model, n=n, l_max=l_max)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.s4(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = x + residual

        return x


class S4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_model,
        d_output,
        n_blocks,
        n,
        l_max,
        dropout = 0.0,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.n_blocks = n_blocks
        
        self.encoder = nn.Linear(d_input, d_model)
        self.blocks = nn.ModuleList([
            S4Block(
                d_model=d_model,
                n=n,
                l_max=l_max,
                dropout=dropout,
            ) for _ in range(n_blocks)
        ])
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, u):
        x = self.encoder(u)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.decoder(x)
        return x