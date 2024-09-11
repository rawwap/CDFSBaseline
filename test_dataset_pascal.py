f"test data pascal"
import torch
from torchvision import transforms
from data.pascal import DatasetPASCAL

# 定义一些基础的转换操作
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 将图片统一调整为 256x256
    transforms.ToTensor(),  # 将图片转换为 tensor
])

# 设置数据路径、fold 和其他参数
datapath = "../../dataset/VOCdevkit"  
fold = 0  # 选择第一个 fold
split = "val"  # 测试验证集
shot = 1  # few-shot 支撑集的数量

# 初始化数据集
dataset = DatasetPASCAL(datapath, fold, transform, split, shot)

# 测试 __len__ 方法
def test_len():
    dataset_length = len(dataset)
    print(f"Dataset length: {dataset_length}")

# 测试 __getitem__ 方法
def test_getitem():
    idx = 0  # 测试第一个样本
    result = dataset[idx]
    support_imgs, support_masks, query_img, query_mask, class_sample, support_names, query_name = result
    print(f"Support images shape: {support_imgs.shape}")
    print(f"Query image shape: {query_img.shape}")
    print(f"Class sample: {class_sample}")
    print(f"Support names: {support_names}")
    print(f"Query name: {query_name}")

# 测试 sample_episode 方法
def test_sample_episode():
    idx = 0  # 测试第一个样本
    query_name, support_names, class_sample = dataset.sample_episode(idx)
    print(f"Query name: {query_name}")
    print(f"Support names: {support_names}")
    print(f"Class sample: {class_sample}")

# 测试 read_img 和 read_mask 方法
def test_read_image_and_mask():
    img_name = "2007_000033"  
    img = dataset.read_img(img_name)
    mask = dataset.read_mask(img_name)
    print(f"Image size: {img.size}")
    print(f"Mask shape: {mask.shape}")

# 测试 extract_ignore_idx 方法
def test_extract_ignore_idx():
    img_name = "2007_000033"  
    mask = dataset.read_mask(img_name)
    class_id = 0  
    mask, boundary = dataset.extract_ignore_idx(mask, class_id)
    print(f"Processed mask shape: {mask.shape}")
    print(f"Boundary shape: {boundary.shape}")

# 运行所有测试
if __name__ == "__main__":
    test_len()               # 测试数据集长度
    test_getitem()           # 测试数据集样本提取
    test_sample_episode()    # 测试支撑集和查询集的采样
    test_read_image_and_mask()  # 测试图像和掩码的读取
    test_extract_ignore_idx()  # 测试掩码的二值化和边界提取
