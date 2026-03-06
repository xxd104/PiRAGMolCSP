import os
import random
from pathlib import Path

# 配置参数
SOURCE_DIR = Path("//home/nyx/N-RGAG/dgl_graphs")
TARGET_DIR = Path("/home/nyx/N-RGAG/dgl_xxx")
SPLIT_RATIO = (0.8, 0.1, 0.1)                   # 划分比例（训练集:测试集:验证集）
RANDOM_SEED = 42                                 # 随机种子（保证可复现性）
FILE_EXTENSION = ".bin"                           # 图文件扩展名（可根据实际情况修改）

def main():
    # 创建目标目录
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图文件（过滤指定扩展名）
    graph_files = [f.name for f in SOURCE_DIR.glob(f"*{FILE_EXTENSION}") if f.is_file()]
    
    if not graph_files:
        raise ValueError("源目录中未找到符合条件的DGL图文件")
    
    # 随机打乱文件列表
    random.seed(RANDOM_SEED)
    random.shuffle(graph_files)
    
    # 计算划分索引
    total = len(graph_files)
    train_idx = int(total * SPLIT_RATIO[0])
    test_idx = train_idx + int(total * SPLIT_RATIO[1])
    
    # 执行划分
    splits = {
        "train": graph_files[:train_idx],
        "test": graph_files[train_idx:test_idx],
        "val": graph_files[test_idx:]
    }
    
    # 验证划分比例
    assert sum(len(ls) for ls in splits.values()) == total, "划分后文件总数不一致"
    print(f"划分结果：训练集{len(splits['train'])}个，测试集{len(splits['test'])}个，验证集{len(splits['val'])}个")
    
    # 保存划分结果
    for name, files in splits.items():
        file_path = TARGET_DIR / f"{name}_list.txt"
        with open(file_path, "w") as f:
            f.write("\n".join(files))
        print(f"{name}集信息已保存至：{file_path}")

if __name__ == "__main__":
    main()