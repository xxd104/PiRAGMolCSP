#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    print(f"\n{description}:")
    print(f"  命令: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        print(f"  返回码: {result.returncode}")
        if result.stdout:
            print(f"  标准输出:")
            for line in result.stdout.split('\n')[:10]:
                if line.strip():
                    print(f"    {line}")
        if result.stderr:
            print(f"  标准错误:")
            for line in result.stderr.split('\n')[:10]:
                if line.strip():
                    print(f"    {line}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  超时")
        return False
    except Exception as e:
        print(f"  错误: {e}")
        return False


def main():
    print("=" * 70)
    print("DFTB+ 诊断工具")
    print("=" * 70)

    dftb_path = "/home/nyx/GRAG/dftbplus/dftbplus-23.1.x86_64-linux/bin/dftb+"
    skf_dir = "/home/nyx/GRAG/dftbplus/slakos/3ob-freq-1-2"

    # 1. 检查文件是否存在
    print(f"\n1. 检查文件:")
    print(f"  DFTB+可执行文件: {dftb_path}")
    if Path(dftb_path).exists():
        print(f"    ✓ 存在")
        # 检查权限
        if os.access(dftb_path, os.X_OK):
            print(f"    ✓ 有执行权限")
        else:
            print(f"    ✗ 没有执行权限，运行: chmod +x {dftb_path}")
    else:
        print(f"    ✗ 不存在")

    print(f"  参数集目录: {skf_dir}")
    if Path(skf_dir).exists():
        print(f"    ✓ 存在")
        # 检查skf文件
        skf_files = list(Path(skf_dir).glob("*.skf"))
        print(f"    找到 {len(skf_files)} 个.skf文件")
        if len(skf_files) > 0:
            print(f"    前5个文件: {[f.name for f in skf_files[:5]]}")
    else:
        print(f"    ✗ 不存在")

    # 2. 检查动态链接库
    print(f"\n2. 检查动态链接库:")
    run_command(f"ldd {dftb_path}", "检查DFTB+依赖库")

    # 3. 测试运行
    print(f"\n3. 测试运行:")
    run_command(f"{dftb_path} --help", "测试--help选项")
    run_command(f"{dftb_path} --version", "测试--version选项")

    # 4. 创建测试输入
    print(f"\n4. 创建测试计算...")
    test_dir = Path("/tmp/dftb_test")
    test_dir.mkdir(exist_ok=True)

    test_input = """Geometry = GenFormat {
  <<< "test.gen"
}

Hamiltonian = DFTB {
  Scc = Yes
  SCCTolerance = 1.0e-5
  MaxSCCIterations = 50
  SlaterKosterFiles = Type2FileNames {
    Prefix = "/home/nyx/GRAG/dftbplus/slakos/3ob-3-1/"
    Separator = "-"
    Suffix = ".skf"
  }
  MaxAngularMomentum {
    H = "s"
  }
}

Options {
  WriteDetailedXML = Yes
}
"""

    test_gen = """1 S
H
1 1 0.0 0.0 0.0
"""

    try:
        with open(test_dir / "test.hsd", "w") as f:
            f.write(test_input)
        with open(test_dir / "test.gen", "w") as f:
            f.write(test_gen)
        print(f"  创建测试文件在: {test_dir}")

        # 运行测试
        os.chdir(test_dir)
        result = subprocess.run([dftb_path], capture_output=True, text=True, timeout=60)
        print(f"  返回码: {result.returncode}")
        if result.stdout:
            print(f"  输出最后10行:")
            for line in result.stdout.split('\n')[-10:]:
                if line.strip():
                    print(f"    {line}")
        if result.stderr:
            print(f"  错误输出:")
            for line in result.stderr.split('\n')[-10:]:
                if line.strip():
                    print(f"    {line}")
    except Exception as e:
        print(f"  测试失败: {e}")

    print(f"\n{'=' * 70}")
    print("诊断完成")
    print("=" * 70)


if __name__ == "__main__":
    main()