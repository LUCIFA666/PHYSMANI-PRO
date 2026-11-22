"""Differentiable batched PoE FK; no Native calls and no implicit GPU choice."""
import numpy as np
import torch
from torch import nn

from .kinematics import PoEKinematics


class TorchPoEKinematics(nn.Module):
    def __init__(self, snapshot):
        super().__init__()
        model = PoEKinematics(snapshot)
        self.register_buffer("q_ref", torch.as_tensor(model.q_ref.copy()))
        self.register_buffer("screws", torch.as_tensor(model.screws.copy()))
        self.register_buffer("reference_tcp", torch.as_tensor(model.reference_tcp.copy()))

    def forward(self, q):
        if (not isinstance(q, torch.Tensor) or not q.is_floating_point() or q.ndim < 1
                or q.shape[-1] != self.q_ref.numel() or not bool(torch.isfinite(q).all())):
            raise ValueError("finite floating joints[...,dof] required")
        reference, screws, tcp = self.q_ref.to(q), self.screws.to(q), self.reference_tcp.to(q)
        identity = torch.eye(3, dtype=q.dtype, device=q.device)
        value = torch.eye(4, dtype=q.dtype, device=q.device).expand(*q.shape[:-1], 4, 4)
        for i in range(reference.numel()):
            v, w = screws[i, :3], screws[i, 3:]
            x, y, z = w.unbind(-1)
            zero = torch.zeros_like(x)
            wx = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
            theta = q[..., i] - reference[i]
            a = theta * torch.sinc(theta / torch.pi)
            b = .5 * theta.square() * torch.sinc(theta / (2*torch.pi)).square()
            c = torch.where(theta.abs() < 1e-4,
                            theta.pow(3)*(1/6-theta.square()/120+theta.pow(4)/5040), theta-theta.sin())
            rotation = identity + a[..., None, None]*wx + b[..., None, None]*(wx@wx)
            position = theta[..., None]*v + b[..., None]*(wx@v) + c[..., None]*(wx@wx@v)
            upper = torch.cat((rotation, position[..., None]), dim=-1)
            bottom = q.new_tensor([0., 0., 0., 1.]).expand(*q.shape[:-1], 1, 4)
            step = torch.cat((upper, bottom), dim=-2)
            value = value @ step
        return value @ tcp
