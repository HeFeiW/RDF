import torch
from torch import nn
import numpy as np

class ParallelSineLayer(nn.Module):
    def __init__(self, num_networks, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.num_networks = num_networks
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features
        
        # 并行权重: (num_networks, out_features, in_features)
        self.weight = nn.Parameter(torch.empty(num_networks, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_networks, out_features))
        else:
            self.register_parameter('bias', None)
            
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.in_features
            else:
                bound = np.sqrt(6 / self.in_features) / self.omega_0
            
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)
    
    def forward(self, x):
        """
        Args:
            x: (N, B, in_features)
        Returns:
            output: (N, B, out_features)
        """
        # 批量矩阵乘法 - 这是关键的并行操作
        output = torch.bmm(x, self.weight.transpose(-1, -2))
        
        if self.bias is not None:
            output += self.bias.unsqueeze(1)
            
        return torch.sin(self.omega_0 * output)

class ParallelSiren(nn.Module):
    def __init__(self, num_networks, in_features, hidden_features, hidden_layers, 
                 out_features, outermost_linear=False, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.num_networks = num_networks
        self.in_features = in_features
        self.out_features = out_features
        
        # 第一层
        self.first_layer = ParallelSineLayer(
            num_networks, in_features, hidden_features, 
            is_first=True, omega_0=first_omega_0
        )
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList([
            ParallelSineLayer(num_networks, hidden_features, hidden_features,
                            is_first=False, omega_0=hidden_omega_0)
            for _ in range(hidden_layers)
        ])
        
        # 输出层
        if outermost_linear:
            self.final_layer = ParallelLinearLayer(num_networks, hidden_features, out_features)
        else:
            self.final_layer = ParallelSineLayer(
                num_networks, hidden_features, out_features,
                is_first=False, omega_0=hidden_omega_0
            )
    
    def forward(self, batched_coords):
        """
        Args:
            batched_coords: (N*B, D) 或 (N, B, D)
        Returns:
            outputs: (N*B, out_features)
        """
        coords = batched_coords.clone().detach().requires_grad_(True)
        if batched_coords.dim() == 2:
            B = batched_coords.shape[0] // self.num_networks
            x = coords.view(self.num_networks, B, self.in_features)
        else:
            x = coords
        # 通过网络层
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.final_layer(x)
        
        # 重塑为 (N*B, out_features)
        return x.view(-1, self.out_features), coords
class ParallelLinearLayer(nn.Module):
    def __init__(self, num_networks, in_features, out_features, bias=True):
        super().__init__()
        self.num_networks = num_networks
        
        self.weight = nn.Parameter(torch.empty(num_networks, out_features, in_features))
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.empty(num_networks, out_features))
        else:
            self.register_parameter('bias', None)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            bound = np.sqrt(6 / self.in_features) / 30  # 使用hidden_omega_0
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)
    
    def forward(self, x):
        output = torch.bmm(x, self.weight.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias.unsqueeze(1)
        return output

# 加载多个模型权重的工具函数
def load_multiple_siren_weights(model_loaded, parallel_siren):
    """
    从多个单独的Siren模型文件加载权重到ParallelSiren
    """
    state_dicts = []
    for model in model_loaded.values():
        state_dicts.append(model['weights'])
    
    # 转换权重格式
    parallel_state_dict = {}
    
    # 处理每一层
    layer_mapping = {
        'net.0': 'first_layer',
        **{f'net.{i+1}': f'hidden_layers.{i}' for i in range(len(parallel_siren.hidden_layers))},
        f'net.{len(parallel_siren.hidden_layers)+1}': 'final_layer'
    }
    for orig_layer, parallel_layer in layer_mapping.items():
        # 确定权重和偏置的键名
        possible_weight_keys = [f'{orig_layer}.linear.weight', f'{orig_layer}.weight']
        possible_bias_keys = [f'{orig_layer}.linear.bias', f'{orig_layer}.bias']
        
        # 找到存在的权重键
        weight_key = None
        for key in possible_weight_keys:
            if key in state_dicts[0]:
                weight_key = key
                break
        
        # 找到存在的偏置键
        bias_key = None
        for key in possible_bias_keys:
            if key in state_dicts[0]:
                bias_key = key
                break
        
        if weight_key:
            weights = torch.stack([sd[weight_key] for sd in state_dicts])
            parallel_state_dict[f'{parallel_layer}.weight'] = weights
        
        if bias_key:
            biases = torch.stack([sd[bias_key] for sd in state_dicts])
            parallel_state_dict[f'{parallel_layer}.bias'] = biases
    
    parallel_siren.load_state_dict(parallel_state_dict)
    print(f"Loaded weights into ParallelSiren with {parallel_siren.num_networks} networks.")
    print('parameter number:', sum(p.numel() for p in parallel_siren.parameters() if p.requires_grad))
    return parallel_siren
