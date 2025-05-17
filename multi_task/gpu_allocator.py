import logging
import torch
from typing import List


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class GPUAllocator:
    def __init__(self, gpu_memory: int, devices: List[int] | str | int):
        """
        Initialize the GPUAllocator with the specified GPU memory and devices.

        Args:
            gpu_memory (int): Amount of GPU memory to allocate, unit: GB.
            devices (List[int]): List of device IDs to allocate memory for.
        """
        self.gpu_memory = gpu_memory
        self.devices = devices
        self.dummy_tensors = {}

    def allocate_gpu_memory(self) -> None:
        """
        Function to allocate GPU memory for the training process.
        """
        if self.gpu_memory > 0:
            tensor_size = self.gpu_memory * 1024 ** 3 // 4  # Convert GB to number of float32 elements
            if isinstance(self.devices, int):
                self.devices = [self.devices]
            elif isinstance(self.devices, str) and self.devices == "auto":
                self.devices = list(range(torch.cuda.device_count()))
            for device in self.devices:
                device_str = f"cuda:{device}"
                self.dummy_tensors[device] = torch.empty(
                    tensor_size, dtype=torch.float32, device=device_str
                )
                logger.info(f"Allocated {self.gpu_memory} GB on device {device_str}")

    def release_gpu_memory(self) -> None:
        """
        Function to release GPU memory after the training process.
        """
        for device, tensor in self.dummy_tensors.items():
            del tensor
            logger.info(f"Released GPU memory on device cuda:{device}")
        self.dummy_tensors.clear()
        torch.cuda.empty_cache()
