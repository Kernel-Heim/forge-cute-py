import torch

from forge_cute_py.ops.reduce_sum import reduce_sum

def main():
    M, N = 4096, 4096
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = reduce_sum(x, dim=-1)
        _ = x.sum(dim=-1)
    torch.cuda.synchronize()
    print("Warmup complete")
    
    # Profile cute
    torch.cuda.nvtx.range_push("cute_reduce_sum")
    for _ in range(100):
        _ = reduce_sum(x, dim=-1)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    # Profile torch
    torch.cuda.nvtx.range_push("torch_sum")
    for _ in range(100):
        _ = x.sum(dim=-1)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

if __name__ == "__main__":
    main()