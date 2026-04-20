import ray
ray.init(ignore_reinit_error=True)
print(ray.available_resources())
print("-" * 80)
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())