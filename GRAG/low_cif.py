import os
import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
import subprocess
import sys
import re

try:
    from xtb.ase.calculator import XTB
    from xtb.interface import XTBException

    HAS_XTB = True
except ImportError:
    print("警告: 未找到xtb包，将使用较简单的EMT计算器")
    from ase.calculators.emt import EMT

    HAS_XTB = False
    XTBException = Exception  

# ====== 配置参数 ======
INPUT_DIR = "/home/nyx/GRAG/generated_cifs/CH4"
OUTPUT_DIR = "/home/nyx/GRAG/low_cifs/CH4"

MAX_STEPS = 1000  # 最大优化步数
FMAX = 0.05  # 力收敛阈值 (eV/Å)
OPTIMIZER = 'BFGS'  # 优化器: 'BFGS' 或 'FIRE'
RELAX_CELL = True  # 是否弛豫晶胞 (True/False)
XTB_METHOD = "GFN2-xTB"  # GFN1-xTB/GFN2-xTB/GFN-FF


# ====== 辅助函数 ======
def read_cif_ignore_spacegroup(cif_path):
    """
    读取CIF文件，忽略空间群信息，直接解析晶胞参数和原子坐标
    优化：暴力定位原子loop，强制读取所有可能的原子行
    """
    with open(cif_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 去除空行和纯注释行，方便解析
    lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

    # 解析晶胞参数
    a = b = c = 0.0
    alpha = beta = gamma = 90.0

    for line in lines:
        if line.startswith('_cell_length_a'):
            a = float(re.split(r'\s+', line)[1])
        elif line.startswith('_cell_length_b'):
            b = float(re.split(r'\s+', line)[1])
        elif line.startswith('_cell_length_c'):
            c = float(re.split(r'\s+', line)[1])
        elif line.startswith('_cell_angle_alpha'):
            alpha = float(re.split(r'\s+', line)[1])
        elif line.startswith('_cell_angle_beta'):
            beta = float(re.split(r'\s+', line)[1])
        elif line.startswith('_cell_angle_gamma'):
            gamma = float(re.split(r'\s+', line)[1])

    # 解析原子信息：暴力定位原子loop
    atoms_data = []
    atom_loop_start_idx = -1  # 原子loop的起始索引

    # 第一步：找到包含_atom_site_type_symbol的loop_（原子数据的核心字段）
    for i, line in enumerate(lines):
        if line.startswith('loop_'):
            # 检查该loop后面是否有原子相关字段
            for j in range(i+1, min(i+10, len(lines))):  # 最多往后查10行
                if '_atom_site_type_symbol' in lines[j]:  # 关键字段：原子符号
                    atom_loop_start_idx = j + 1  # 原子数据从字段定义后开始
                    break
            if atom_loop_start_idx != -1:
                break

    # 第二步：从原子loop起始位置开始，读取所有原子数据行
    if atom_loop_start_idx != -1:
        print(f"  找到原子数据loop，从第{atom_loop_start_idx}行开始读取")
        for line in lines[atom_loop_start_idx:]:
            # 停止条件：遇到下一个loop_或新的_字段（非原子相关）
            if line.startswith('loop_') or (line.startswith('_') and not line.startswith('_atom_site_')):
                break
            # 分割字段
            parts = re.split(r'\s+', line)
            # 必须至少有：label + symbol + x + y + z（5个字段）
            if len(parts) >= 5:
                try:
                    # 提取原子符号和分数坐标（兼容不同字段顺序，优先按位置）
                    symbol = parts[1]  # _atom_site_type_symbol在第二个字段
                    x = float(parts[2])  # _atom_site_fract_x
                    y = float(parts[3])  # _atom_site_fract_y
                    z = float(parts[4])  # _atom_site_fract_z
                    # 只保留有效元素
                    if symbol in ['C', 'H', 'O', 'N'] and not (np.isnan([x, y, z]).any()):
                        atoms_data.append((symbol, x, y, z))
                except (ValueError, IndexError) as e:
                    # 跳过无效行，不中断程序
                    continue

    # 备用方案：如果没找到loop，直接扫描所有行找原子数据
    if not atoms_data:
        print("  未找到原子loop，尝试全局扫描原子数据")
        for line in lines:
            if line.startswith(('_', 'loop_')):
                continue
            parts = re.split(r'\s+', line)
            # 匹配两种格式：
            # 格式1：label symbol x y z ...（parts[1]是符号）
            # 格式2：symbol x y z ...（parts[0]是符号）
            if len(parts) >= 4:
                try:
                    if parts[1] in ['C', 'H', 'O', 'N']:  # 格式1
                        symbol = parts[1]
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    elif parts[0] in ['C', 'H', 'O', 'N']:  # 格式2
                        symbol = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    else:
                        continue
                    if not (np.isnan([x, y, z]).any()):
                        atoms_data.append((symbol, x, y, z))
                except (ValueError, IndexError) as e:
                    continue

    # 验证读取结果
    if not atoms_data:
        raise ValueError(f"未找到原子数据！请检查CIF格式")
    print(f"  成功读取到 {len(atoms_data)} 个原子")

    # 创建Atoms对象（分数坐标转笛卡尔坐标）
    symbols = [data[0] for data in atoms_data]
    frac_positions = [[data[1], data[2], data[3]] for data in atoms_data]

    # 设置晶胞（三斜晶胞公式）
    cell = np.array([
        [a, 0.0, 0.0],
        [b * np.cos(np.radians(gamma)), b * np.sin(np.radians(gamma)), 0.0],
        [
            c * np.cos(np.radians(beta)),
            c * (np.cos(np.radians(alpha)) - np.cos(np.radians(beta)) * np.cos(np.radians(gamma))) / np.sin(
                np.radians(gamma)),
            c * np.sqrt(
                1 - np.cos(np.radians(alpha)) ** 2 - np.cos(np.radians(beta)) ** 2 - np.cos(np.radians(gamma)) ** 2 +
                2 * np.cos(np.radians(alpha)) * np.cos(np.radians(beta)) * np.cos(np.radians(gamma))
            ) / np.sin(np.radians(gamma))
        ]
    ])

    # 确保晶胞矩阵有效
    cell = np.nan_to_num(cell, nan=0.0, posinf=0.0, neginf=0.0)
    # 处理退化晶胞（避免全零）
    if np.allclose(cell, 0.0):
        raise ValueError("晶胞参数无效，全为零")

    # 分数坐标转换为笛卡尔坐标
    cart_positions = np.dot(frac_positions, cell)

    atoms = Atoms(symbols=symbols, positions=cart_positions, cell=cell, pbc=True)

    return atoms


# ====== 主程序 ======
def relax_structure(input_path, output_path):
    """弛豫单个结构"""
    try:
        print(f"正在处理: {os.path.basename(input_path)}")

        # 读取CIF文件（忽略空间群）
        atoms = read_cif_ignore_spacegroup(input_path)
        original_atoms = atoms.copy()

        print(f"  原子数: {len(atoms)}")
        cell_lengths = atoms.cell.lengths()
        cell_angles = atoms.cell.angles()
        print(f"  晶胞: [{cell_lengths[0]:.3f}, {cell_lengths[1]:.3f}, {cell_lengths[2]:.3f}] Å, "
              f"角度: [{cell_angles[0]:.3f}, {cell_angles[1]:.3f}, {cell_angles[2]:.3f}]°")

        # 设置计算器（容错机制）
        calculator = None
        if HAS_XTB:
            try:
                # 优先使用GFN1-xTB（小分子通用）
                calculator = XTB(method=XTB_METHOD)
                print(f"  使用xtb {XTB_METHOD} 方法进行计算")
            except XTBException as e:
                print(f"  {XTB_METHOD} 方法失败: {str(e)}")
                try:
                    # 备选：GFN2-xTB
                    calculator = XTB(method="GFN2-xTB")
                    print("  切换到 GFN2-xTB 方法")
                except XTBException as e2:
                    print(f"  GFN2-xTB 方法也失败: {str(e2)}")
                    # 最终切换到EMT
                    calculator = EMT()
                    print("  切换到 EMT 计算器 (警告: 对有机分子精度有限)")
        else:
            # 直接使用EMT
            calculator = EMT()
            print("  使用EMT计算器 (警告: 对有机分子精度有限)")

        atoms.calc = calculator
        original_atoms.calc = calculator  # 用于计算初始能量

        # 设置优化器
        if OPTIMIZER.lower() == 'bfgs':
            optimizer_class = BFGS
        else:
            from ase.optimize import FIRE
            optimizer_class = FIRE

        # 选择是否弛豫晶胞
        if RELAX_CELL:
            # 弛豫原子位置和晶胞
            ucf = UnitCellFilter(atoms)
            optimizer = optimizer_class(ucf, trajectory=None)
            print("  弛豫模式: 原子位置 + 晶胞")
        else:
            # 只弛豫原子位置，固定晶胞
            optimizer = optimizer_class(atoms, trajectory=None)
            print("  弛豫模式: 仅原子位置")

        # 运行几何优化
        print("  开始优化...")
        optimizer.run(fmax=FMAX, steps=MAX_STEPS)
        print("  优化完成")

        # 计算能量变化
        try:
            initial_energy = original_atoms.get_potential_energy()
            final_energy = atoms.get_potential_energy()
            energy_change = final_energy - initial_energy
            print(f"  初始能量: {initial_energy:.6f} eV")
            print(f"  最终能量: {final_energy:.6f} eV")
            print(f"  能量变化: {energy_change:.6f} eV")
        except Exception as e:
            print(f"  能量计算警告: {str(e)}")

        # 保存弛豫后的结构
        write(output_path, atoms)
        print(f"  成功保存到: {os.path.basename(output_path)}")

        return True

    except Exception as e:
        print(f"  错误: {str(e)}")
        # 打印详细的错误信息用于调试
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录不存在: {INPUT_DIR}")
        return

    # 获取所有CIF文件
    cif_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.cif')]

    if not cif_files:
        print(f"在 {INPUT_DIR} 中未找到CIF文件")
        return

    print(f"找到 {len(cif_files)} 个CIF文件")
    print("=" * 50)

    # 处理每个文件
    success_count = 0
    for filename in cif_files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_filename = filename.replace('.cif', '_relaxed.cif')
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        if relax_structure(input_path, output_path):
            success_count += 1

        print("-" * 30)

    print(f"处理完成! 成功: {success_count}/{len(cif_files)}")


if __name__ == "__main__":
    main()