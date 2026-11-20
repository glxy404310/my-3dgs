"""Seeded CPU nearest-neighbor and CUDA dispatch witness; no research inputs."""
import numpy as np
import scipy
from scipy.spatial import cKDTree
import torch
import diff_gaussian_rasterization
import simple_knn
import fused_ssim

assert np.__version__ == '2.2.6' and scipy.__version__ == '1.15.3'
points = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
query = np.array([[.1, .2, 0.], [1.9, .1, 0.], [0., 2.8, 0.]])
distance, index = cKDTree(points).query(query, workers=1)
assert np.array_equal(index, [0, 1, 2])
assert np.allclose(distance, np.linalg.norm(query - points[index], axis=1), rtol=1e-14, atol=1e-14)
torch.manual_seed(0)
x = torch.randn(8, 8, device='cuda')
y = x @ x
assert torch.isfinite(y).all() and y.numel() == 64
torch.cuda.synchronize()
print('TRANSFER_ENV_WITNESS', scipy.__version__, index.tolist(), tuple(y.shape), torch.cuda.get_device_name(0), float(y.sum()))
