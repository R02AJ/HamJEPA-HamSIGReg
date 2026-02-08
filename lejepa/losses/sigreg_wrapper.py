import torch
import torch.nn as nn

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None


class SIGReg(nn.Module):
    """
    LeJEPA-style SIGReg (Epps-Pulley CF distance) supporting [N,K] or [V,B,K].
    - directions seeded by global_step
    - cached t-grid / exp_f buffers
    - DDP-friendly all-reduce on complex ECF
    """

    def __init__(self, num_slices: int = 256, t_min: float = -5.0, t_max: float = 5.0, num_t: int = 17):
        super().__init__()
        self.num_slices = int(num_slices)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.num_t = int(num_t)

        # cache on CPU; cast in forward
        t_cpu = torch.linspace(self.t_min, self.t_max, self.num_t, dtype=torch.float32)
        exp_f_cpu = torch.exp(-0.5 * t_cpu**2)
        self.register_buffer("_t_cpu", t_cpu, persistent=True)
        self.register_buffer("_exp_f_cpu", exp_f_cpu, persistent=True)

    @staticmethod
    def _world_size() -> int:
        if dist is not None and dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def _all_reduce_mean_real(x: torch.Tensor) -> torch.Tensor:
        if dist is None or (not dist.is_available()) or (not dist.is_initialized()):
            return x
        x = x.contiguous()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x / float(dist.get_world_size())

    def _t_and_exp_f(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._t_cpu.to(device=device, dtype=dtype)
        exp_f = self._exp_f_cpu.to(device=device, dtype=dtype)
        return t, exp_f

    def _sample_A(self, k: int, device: torch.device, dtype: torch.dtype, global_step: int) -> torch.Tensor:
        g = torch.Generator(device=device)
        g.manual_seed(int(global_step))
        A = torch.randn((k, self.num_slices), generator=g, device=device, dtype=dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        return A

    def forward(self, x: torch.Tensor, global_step: int = 0) -> torch.Tensor:
        """
        x: [N,K] or [V,B,K]
        returns: scalar loss
        """
        if x.dim() == 2:
            n, k = x.shape
            if n < 2:
                return torch.zeros([], device=x.device, dtype=x.dtype)
            t, exp_f = self._t_and_exp_f(x.device, x.dtype)
            A = self._sample_A(k, x.device, x.dtype, global_step)
            proj = x @ A  # [N,M]
            x_t = proj.unsqueeze(-1) * t  # [N,M,T]
            c = torch.cos(x_t).mean(dim=0)  # [M,T]
            s = torch.sin(x_t).mean(dim=0)  # [M,T]
            c = self._all_reduce_mean_real(c)
            s = self._all_reduce_mean_real(s)
            err = ((c - exp_f) ** 2 + (s ** 2)) * exp_f  # [M,T]
            N_total = n * self._world_size()
            stat = torch.trapz(err, t, dim=-1) * float(N_total)  # [M]
            return stat.mean()

        if x.dim() == 3:
            v, b, k = x.shape
            if b < 2:
                return torch.zeros([], device=x.device, dtype=x.dtype)
            t, exp_f = self._t_and_exp_f(x.device, x.dtype)
            A = self._sample_A(k, x.device, x.dtype, global_step)
            x_flat = x.reshape(v * b, k)
            proj = (x_flat @ A).view(v, b, self.num_slices)  # [V,B,M]
            x_t = proj.unsqueeze(-1) * t                      # [V,B,M,T]
            c = torch.cos(x_t).mean(dim=1)  # [V,M,T]
            s = torch.sin(x_t).mean(dim=1)  # [V,M,T]
            c = self._all_reduce_mean_real(c)
            s = self._all_reduce_mean_real(s)
            err = ((c - exp_f) ** 2 + (s ** 2)) * exp_f       # [V,M,T]
            N_total = b * self._world_size()
            stat = torch.trapz(err, t, dim=-1) * float(N_total)  # [V,M]
            return stat.mean()

        raise ValueError(f"SIGReg expects x shape [N,K] or [V,B,K], got {tuple(x.shape)}")


# Backwards-compatibility alias
SlicingGaussianReg = SIGReg
