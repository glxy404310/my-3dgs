# Local WSL compute ledger

### env: 3dgs@caff637c

- how: Python virtual environment `/home/administrator/venvs/3dgs` in WSL2 distribution `Ubuntu-22.04`
- tier: `{cpus: local, mem_gib: local, gpus: 1x NVIDIA GeForce RTX 5070 12227 MiB}`
- repository: `/home/administrator/projects/gaussian-splatting` at `54c035f7834b564019656c3e3fcc3646292f727d`
- data: `/home/administrator/datasets/tandt_db/tandt/truck`
- validated: 2026-10-21, tier 1 imports and tier 2 seeded CUDA witness passed
- current GPU preflight: 2579 MiB used, 12227 MiB total, no compute process reported; graphical Windows session is active
- gotcha: `nvcc` is not on the default non-interactive PATH; launchers export `CUDA_HOME=/usr/local/cuda-12.8` and prepend `$CUDA_HOME/bin`

Validation invocation:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
/home/administrator/venvs/3dgs/bin/python -c 'import torch; import diff_gaussian_rasterization, simple_knn, fused_ssim; torch.manual_seed(0); x=torch.randn(8,8,device="cuda"); y=x@x; print("WITNESS", tuple(y.shape), torch.cuda.get_device_name(0), float(y.sum()))'
```

Expected prefix:

```text
WITNESS (8, 8) NVIDIA GeForce RTX 5070
```

Experiment launcher:

```bash
bash "/mnt/d/Documents/my 3dgs/research/experiments/launchers/run-truck-diag-7k.sh"
```

The launcher trains truck from scratch to 7000 iterations at resolution 1, evaluates at 7000, saves the point cloud, and writes `chkpnt7000.pth`. It writes to a new model directory and does not modify earlier truck outputs.

