import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    # 你也可以检查 NCCL 版本 (如果 fbgemm-gpu 或其依赖项需要)
    # print(f"PyTorch NCCL Version: {torch.cuda.nccl.version()}")

# 尝试导入 fbgemm_gpu 相关的包 (如果它被设计为需要显式导入以注册操作)
# 有时，fbgemm 的操作是通过像 torchrec 这样的上层库间接导入和注册的
# 例如:
# try:
#     import torchrec
#     print("TorchRec imported successfully, which might register FBGEMM ops.")
# except ImportError:
#     print("TorchRec not found or not imported.")
#
# try:
#     import fbgemm_gpu
#     print("fbgemm_gpu imported successfully.")
# except ImportError:
#     print("fbgemm_gpu package itself could not be imported directly (this might be normal depending on setup).")


print("\nAttributes in torch.ops.fbgemm:")
fbgemm_ops = dir(torch.ops.fbgemm)
print(fbgemm_ops)

if 'asynchronous_complete_cumsum' in fbgemm_ops:
    print("\n'asynchronous_complete_cumsum' IS found in torch.ops.fbgemm!")
    try:
        op = torch.ops.fbgemm.asynchronous_complete_cumsum
        print("Successfully accessed torch.ops.fbgemm.asynchronous_complete_cumsum.")
    except AttributeError as e:
        print(f"Error accessing attribute even if listed: {e}") # 不太可能发生，但作为完整性检查
else:
    print("\nERROR: 'asynchronous_complete_cumsum' is NOT found in torch.ops.fbgemm.")
    print("This indicates that fbgemm-gpu operations were not loaded correctly.")
    print("Please check your fbgemm-gpu installation, compatibility with PyTorch and CUDA versions, and ensure it was compiled correctly.")