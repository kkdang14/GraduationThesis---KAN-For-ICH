import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class WendlandC2RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 4.,
        num_grids: int = 10,
        bandwidth: float = None,  #Nếu None, sẽ được tính toán dựa trên khoảng cách giữa các tâm lưới
        learnable_centers: bool = False,
        learnable_bandwidth: bool = False,
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        
        # Khởi tạo tâm lưới dạng đều
        grid = torch.linspace(grid_min, grid_max, num_grids)
        if learnable_centers:
            self.grid = torch.nn.Parameter(grid, requires_grad=True)
        else:
            self.register_buffer('grid', grid)
        
        # Khơi tạo bandwidth
        if bandwidth is None:
            bandwidth = (grid_max - grid_min) / (num_grids - 1) * 1.5  
        
        if learnable_bandwidth:
            self.bandwidth = torch.nn.Parameter(torch.tensor(bandwidth), requires_grad=True)
        else:
            self.register_buffer('bandwidth', torch.tensor(bandwidth))

    def wendland_c2(self, r):
        # Wendland_C2 kernel: (1-r)^4_+ * (4r + 1)
        #Chỉ bằng 0 khi r > 1 (compact support)
        #Ở đây r = |x - c| / h, với c là center, h là bandwidth
        #Ở đây ta tính r bên ngoài để tận dụng broadcasting

        r = torch.clamp(r, min=0.0)
        mask = (r <= 1.0).float()
        
        one_minus_r = torch.clamp(1.0 - r, min=0.0)
        result = (one_minus_r ** 4) * (4 * r + 1) * mask
        
        return result

    def forward(self, x):
        # Tính r
        distances = torch.abs(x[..., None] - self.grid) 
        r = distances / torch.abs(self.bandwidth)  
        
        basis_values = self.wendland_c2(r)  
        
        return basis_values


class MochiKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
        mochi_bandwidth: float = None,
        learnable_centers: bool = False,
        learnable_bandwidth: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
            
        # Khởi tạo Wendland C2 RBF
        self.rbf = WendlandC2RadialBasisFunction(
            grid_min, grid_max, num_grids, 
            bandwidth=mochi_bandwidth,
            learnable_centers=learnable_centers,
            learnable_bandwidth=learnable_bandwidth
        )
        
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        '''Plot learned curves using mochi C2 basis'''
        ng = self.rbf.num_grids
        h = self.rbf.bandwidth.item() if torch.is_tensor(self.rbf.bandwidth) else self.rbf.bandwidth
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]
        
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )
        
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y


class MochiKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
        mochi_bandwidth: float = None,
        learnable_centers: bool = False,
        learnable_bandwidth: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MochiKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
                mochi_bandwidth=mochi_bandwidth,
                learnable_centers=learnable_centers,
                learnable_bandwidth=learnable_bandwidth,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_sparsity_stats(self):
        """Analyze sparsity due to compact support"""
        total_activations = 0
        active_activations = 0
        
        for layer in self.layers:
            # This would need to be called during forward pass to get actual sparsity
            pass
        
        return {
            'theoretical_sparsity': 'Compact support reduces computations',
            'bandwidth_per_layer': [layer.rbf.bandwidth.item() if torch.is_tensor(layer.rbf.bandwidth) 
                                    else layer.rbf.bandwidth for layer in self.layers]
        }

