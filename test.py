import torch
import torch.nn as nn
import numpy as np
import time
from ParallelSiren import ParallelSiren, load_multiple_siren_weights


def benchmark_inference():
    # 参数设置
    model_path = '/workspace/RDF/siren_model.pth' 
    model_loaded = torch.load(model_path,map_location='cpu')
    num_networks = len(model_loaded)  # 假设模型文件中包含多个网络
    print('num_networks:',num_networks)

    batch_size = 1024
    in_features = 3
    
    # 创建并行模型
    parallel_siren = ParallelSiren(
        num_networks=num_networks,
        in_features=in_features,
        hidden_features=256,
        hidden_layers=3,
        out_features=1,
        outermost_linear=True
    ).cuda()
    
    # 假设你有多个训练好的模型权重文件
    parallel_siren = load_multiple_siren_weights(model_loaded, parallel_siren)
    
    # 准备测试数据
    test_coords = torch.randn(num_networks * batch_size, in_features).cuda()
    
    # 性能测试
    import time
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = parallel_siren(test_coords)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            outputs = parallel_siren(test_coords)
    
    torch.cuda.synchronize()
    end_time = time.time()
    print(f'输入形状: {num_networks} networks * batch size {batch_size} * input dim {in_features}')
    print(f'输出形状: {num_networks * batch_size} * output dim 1')
    print(f"并行推理时间: {(end_time - start_time) / 100 * 1000:.2f} ms")
def compare_performance():
    """对比循环调用 vs 真正并行的性能"""
    from Siren import Siren
    # 参数设置
    model_path = '/workspace/RDF/siren_model.pth' 
    model_loaded = torch.load(model_path,map_location='cpu')
    num_networks = len(model_loaded)  # 假设模型文件中包含多个网络
    print('num_networks:',num_networks)
    batch_size = 1024
    in_features = 3
    out_features = 1
    
    # 创建单独的模型（循环方式）
    individual_models = [
        Siren(in_features=in_features, out_features=out_features, hidden_features=256, 
              hidden_layers=3, outermost_linear=True).cuda()
        for _ in range(num_networks)
    ]
    
    # 创建并行模型
    parallel_model = ParallelSiren(
        num_networks=num_networks, in_features=3, out_features=1,
        hidden_features=256, hidden_layers=3, outermost_linear=True
    ).cuda()
    
    # 加载权重
    for i, model in enumerate(individual_models):
        model.load_state_dict(model_loaded[list(model_loaded.keys())[i]]['weights'])
        model.eval()
    parallel_model = load_multiple_siren_weights(model_loaded, parallel_model)
    parallel_model.eval()
    
    test_data = torch.randn(num_networks, batch_size, 3).cuda()
    # 预热
    with torch.no_grad():
        for _ in range(10):
            for i, model in enumerate(individual_models):
                _ = model(test_data[i])
            _ = parallel_model(test_data)
            
    # 测试循环方式
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            outputs_loop = []
            for i, model in enumerate(individual_models):
                output, _ = model(test_data[i])
                outputs_loop.append(output)
    torch.cuda.synchronize()
    loop_time = time.time() - start
    
    # 测试并行方式
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            outputs_parallel = parallel_model(test_data)
    torch.cuda.synchronize()
    parallel_time = time.time() - start
    # 验证结果一致性
    outputs_loop = torch.cat(outputs_loop, dim=0)
    print('outputs_loop shape:', outputs_loop) 
    assert torch.allclose(outputs_loop, outputs_parallel, atol=1e-5), "Outputs do not match!"
    print(f'输入形状: {num_networks} networks * batch size {batch_size} * input dim {in_features}')
    print(f'输出形状: {num_networks * batch_size} * output dim {out_features}')
    print(f"循环方式: {loop_time/100*1000:.2f} ms")
    print(f"并行方式: {parallel_time/100*1000:.2f} ms")
    print(f"加速比: {loop_time/parallel_time:.2f}x")
if __name__ == '__main__':
    benchmark_inference()
    compare_performance()