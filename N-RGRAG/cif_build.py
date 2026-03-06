import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from scipy import stats
from matplotlib import rcParams
from collections import defaultdict

# ======================== 1. 全局配置 ========================
# 可视化样式与字体设置（
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['DejaVu Sans']  # 有中文需求替换为 ['SimHei']
rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300  # 高清图片
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)

# 核心路径定义
SOURCE_ROOT = "/home/nyx/N-RGAG/zxs_gen_detailed/zxs/dftb+/done"
OUTPUT_CIF_DIR = "/home/nyx/N-RGAG/raw_cifs"
OUTPUT_VIS_DIR = "/home/nyx/N-RGAG/raw_vis"

# 创建输出目录（不存在则自动创建）
for dir_path in [OUTPUT_CIF_DIR, OUTPUT_VIS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 已创建目录: {dir_path}")


# ======================== 2. 数据提取核心函数 ========================
def extract_detailed_data(detailed_file_path):
    """
    从detailed.out提取关键数据：离子受力、应力张量、总能量
    返回值：字典（包含所有提取数据，失败返回None）
    """
    extract_result = {
        "ion_forces": [],  # 离子受力 (n, 3) 数组 [x, y, z]
        "stress_tensor": [],  # 应力张量 (3, 3) 数组
        "total_energy_H": None,  # 总能量（哈特里）
        "total_energy_eV": None,  # 总能量（电子伏特）
        "force_magnitudes": []  # 每个离子受力的总大小（√(x²+y²+z²)）
    }

    # 校验文件是否存在
    if not os.path.exists(detailed_file_path):
        print(f"❌ 警告：detailed.out文件不存在 - {detailed_file_path}")
        return None

    try:
        with open(detailed_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # ------------- 提取离子受力 -------------
        forces_pattern = r"Total Forces\n((?:\s*\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\n)+)"
        forces_match = re.search(forces_pattern, content)
        if forces_match:
            forces_lines = forces_match.group(1).strip().split('\n')
            for line in forces_lines:
                # 提取所有数值（跳过编号，取后3列）
                nums = re.findall(r"[-+]?\d+\.\d+", line)
                if len(nums) >= 3:
                    force_xyz = [float(n) for n in nums[-3:]]
                    extract_result["ion_forces"].append(force_xyz)
                    # 计算受力总大小
                    magnitude = np.sqrt(np.sum(np.square(force_xyz)))
                    extract_result["force_magnitudes"].append(magnitude)
        else:
            print(f"⚠️ 警告：{detailed_file_path} 未找到离子受力数据")

        # ------------- 提取应力张量 -------------
        stress_pattern = r"Total stress tensor\n((?:\s*[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\n){3})"
        stress_match = re.search(stress_pattern, content)
        if stress_match:
            stress_lines = stress_match.group(1).strip().split('\n')[:3]  # 确保只取3行
            for line in stress_lines:
                nums = re.findall(r"[-+]?\d+\.\d+", line)
                if len(nums) >= 3:
                    extract_result["stress_tensor"].append([float(n) for n in nums[:3]])
        else:
            print(f"⚠️ 警告：{detailed_file_path} 未找到应力张量数据")

        # ------------- 提取总能量 -------------
        energy_pattern = r"Total energy:\s+([-+]?\d+\.\d+)\s+H\s+([-+]?\d+\.\d+)\s+eV"
        energy_match = re.search(energy_pattern, content)
        if energy_match:
            extract_result["total_energy_H"] = float(energy_match.group(1))
            extract_result["total_energy_eV"] = float(energy_match.group(2))
        else:
            print(f"⚠️ 警告：{detailed_file_path} 未找到总能量数据")

        # 数据校验
        if len(extract_result["stress_tensor"]) != 3:
            print(f"⚠️ 警告：{detailed_file_path} 应力张量数据不完整（仅{len(extract_result['stress_tensor'])}行）")

        return extract_result

    except Exception as e:
        print(f"❌ 错误：处理{detailed_file_path}失败 - {str(e)}")
        return None


# ======================== 3. CIF文件处理函数 ========================
def extract_elements_from_cif(cif_file_path):
    """从CIF文件提取元素组成（返回元素列表）"""
    elements = []
    if not os.path.exists(cif_file_path):
        return elements

    try:
        with open(cif_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # CIF文件元素行特征：包含元素符号+坐标的行（适配常见CIF格式）
        element_pattern = r"^\s*([A-Z][a-z]?)\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+"
        for line in lines:
            match = re.match(element_pattern, line.strip())
            if match:
                elements.append(match.group(1))
    except Exception as e:
        print(f"⚠️ 警告：提取{CIF文件路径}元素失败 - {str(e)}")

    return elements


def process_single_subdir(subdir_path):
    """
    处理单个子目录：提取数据 → 追加到CIF → 保存新CIF
    返回值：(是否成功, 子目录名, 提取的统计数据)
    """
    # 关键文件路径
    detailed_path = os.path.join(subdir_path, "detailed.out")
    original_cif_path = os.path.join(subdir_path, "geom.out.cif")
    subdir_name = os.path.basename(subdir_path)
    output_cif_path = os.path.join(OUTPUT_CIF_DIR, f"{subdir_name}.cif")

    # 校验关键文件是否存在
    if not os.path.exists(detailed_path) or not os.path.exists(original_cif_path):
        print(f"❌ 跳过：{subdir_path} 缺少detailed.out或geom.out.cif")
        return (False, subdir_name, None)

    # 提取detailed.out数据
    detailed_data = extract_detailed_data(detailed_path)
    if not detailed_data:
        return (False, subdir_name, None)

    # 提取CIF文件元素信息
    elements = extract_elements_from_cif(original_cif_path)
    element_counts = defaultdict(int)
    for elem in elements:
        element_counts[elem] += 1

    # 读取原始CIF内容
    try:
        with open(original_cif_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_cif_content = f.read()

        # 构造追加内容（用#注释，完全不影响CIF原有格式）
        append_content = "\n\n# ========== DFTB+ Calculation Results (Auto-Generated) ==========\n"

        # 1. 总能量
        append_content += f"# Total Energy (Hartree): {detailed_data['total_energy_H']}\n"
        append_content += f"# Total Energy (eV): {detailed_data['total_energy_eV']}\n"

        # 2. 离子受力（按编号）
        append_content += "# Ion Forces (x, y, z) [atomic units]\n"
        for idx, force in enumerate(detailed_data["ion_forces"], 1):
            append_content += f"# Force_{idx:03d}: {force[0]:.12f} {force[1]:.12f} {force[2]:.12f}\n"

        # 3. 应力张量
        append_content += "# Stress Tensor (3x3) [atomic units]\n"
        for row_idx, tensor_row in enumerate(detailed_data["stress_tensor"], 1):
            append_content += f"# Stress_Row_{row_idx}: {tensor_row[0]:.12f} {tensor_row[1]:.12f} {tensor_row[2]:.12f}\n"

        # 4. 受力统计
        if detailed_data["force_magnitudes"]:
            avg_force = np.mean(detailed_data["force_magnitudes"])
            max_force = np.max(detailed_data["force_magnitudes"])
            append_content += f"# Avg Ion Force Magnitude: {avg_force:.12f}\n"
            append_content += f"# Max Ion Force Magnitude: {max_force:.12f}\n"

        # 5. 元素组成
        append_content += "# Element Composition: " + ", ".join([f"{k}({v})" for k, v in element_counts.items()]) + "\n"

        # 合并并保存新CIF
        new_cif_content = original_cif_content + append_content
        with open(output_cif_path, 'w', encoding='utf-8') as f:
            f.write(new_cif_content)

        # 整理统计数据（用于后续可视化）
        stats_data = {
            "subdir_name": subdir_name,
            "elements": elements,
            "element_counts": dict(element_counts),
            "total_energy_H": detailed_data["total_energy_H"],
            "total_energy_eV": detailed_data["total_energy_eV"],
            "avg_force_magnitude": np.mean(detailed_data["force_magnitudes"]) if detailed_data[
                "force_magnitudes"] else None,
            "max_force_magnitude": np.max(detailed_data["force_magnitudes"]) if detailed_data[
                "force_magnitudes"] else None,
            "stress_tensor": detailed_data["stress_tensor"],
            "stress_trace": np.trace(detailed_data["stress_tensor"]) if len(
                detailed_data["stress_tensor"]) == 3 else None  # 应力张量迹
        }

        print(f"✅ 成功生成CIF: {output_cif_path}")
        return (True, subdir_name, stats_data)

    except Exception as e:
        print(f"❌ 错误：处理{subdir_path}的CIF文件失败 - {str(e)}")
        return (False, subdir_name, None)


def batch_process_all_dirs():
    """批量遍历所有子目录，执行数据提取和CIF生成"""
    all_stats_data = []  # 存储所有目录的统计数据（用于可视化）
    processed_count = 0
    failed_count = 0

    # 递归遍历所有子目录
    print("\n========== 开始批量处理目录 ==========")
    for root, dirs, files in os.walk(SOURCE_ROOT):
        # 仅处理包含detailed.out和geom.out.cif的目录
        if "detailed.out" in files and "geom.out.cif" in files:
            success, subdir_name, stats_data = process_single_subdir(root)
            if success:
                processed_count += 1
                if stats_data:
                    all_stats_data.append(stats_data)
            else:
                failed_count += 1

    print(f"\n========== 处理完成 ==========")
    print(f"✅ 成功处理: {processed_count} 个目录")
    print(f"❌ 处理失败: {failed_count} 个目录")
    return all_stats_data


# ======================== 4. 多维度可视化函数 ========================
def generate_visualizations(all_stats_data):
    """基于所有统计数据生成多维度可视化图表"""
    if not all_stats_data:
        print("❌ 无有效数据，跳过可视化")
        return

    # 转换为DataFrame（方便数据处理）
    df = pd.DataFrame(all_stats_data)
    # 过滤无效数据
    df = df.dropna(subset=["total_energy_eV", "avg_force_magnitude"])
    if df.empty:
        print("❌ 过滤后无有效数据，跳过可视化")
        return

    print("\n========== 开始生成可视化图表 ==========")

    # ------------- 可视化1：元素分布（柱状图）-------------
    plt.figure()
    element_counter = defaultdict(int)
    for elem_counts in df["element_counts"]:
        for elem, count in elem_counts.items():
            element_counter[elem] += count
    elements = list(element_counter.keys())
    counts = list(element_counter.values())

    sns.barplot(x=elements, y=counts, palette="viridis")
    plt.title("Element Distribution Across All Systems")
    plt.xlabel("Element")
    plt.ylabel("Total Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, "element_distribution.png"))
    plt.close()
    print("✅ 生成：element_distribution.png")

    # ------------- 可视化2：总能量分布（eV，直方图+核密度）-------------
    plt.figure()
    sns.histplot(df["total_energy_eV"], kde=True, color="steelblue", bins=20)
    plt.title("Total Energy Distribution (eV)")
    plt.xlabel("Total Energy (eV)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, "energy_distribution_ev.png"))
    plt.close()
    print("✅ 生成：energy_distribution_ev.png")

    # ------------- 可视化3：平均受力大小分布（箱线图+小提琴图）-------------
    plt.figure()
    sns.violinplot(y=df["avg_force_magnitude"], color="lightgreen", inner="box")
    plt.title("Distribution of Average Ion Force Magnitude")
    plt.ylabel("Average Force Magnitude (atomic units)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, "avg_force_distribution.png"))
    plt.close()
    print("✅ 生成：avg_force_distribution.png")

    # ------------- 可视化4：总能量与平均受力的相关性散点图 -------------
    plt.figure()
    sns.scatterplot(x=df["avg_force_magnitude"], y=df["total_energy_eV"], alpha=0.7, color="crimson")
    # 计算相关系数
    corr = stats.pearsonr(df["avg_force_magnitude"], df["total_energy_eV"])[0]
    plt.title(f"Correlation Between Avg Force and Total Energy (r={corr:.3f})")
    plt.xlabel("Average Force Magnitude (atomic units)")
    plt.ylabel("Total Energy (eV)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, "energy_force_correlation.png"))
    plt.close()
    print("✅ 生成：energy_force_correlation.png")

    # ------------- 可视化5：主要元素的能量分布（箱线图）-------------
    # 筛选出现次数最多的前5个元素
    top_elements = [elem for elem, _ in sorted(element_counter.items(), key=lambda x: x[1], reverse=True)[:5]]
    # 构造元素-能量映射
    elem_energy_data = []
    for idx, row in df.iterrows():
        for elem in top_elements:
            if elem in row["element_counts"]:
                elem_energy_data.append({
                    "element": elem,
                    "total_energy_eV": row["total_energy_eV"]
                })
    if elem_energy_data:
        elem_energy_df = pd.DataFrame(elem_energy_data)
        plt.figure()
        sns.boxplot(x="element", y="total_energy_eV", data=elem_energy_df, palette="Set2")
        plt.title("Total Energy Distribution by Major Elements")
        plt.xlabel("Element")
        plt.ylabel("Total Energy (eV)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_VIS_DIR, "element_energy_boxplot.png"))
        plt.close()
        print("✅ 生成：element_energy_boxplot.png")

    # ------------- 可视化6：应力张量迹的分布 -------------
    stress_trace_data = df["stress_trace"].dropna()
    if not stress_trace_data.empty:
        plt.figure()
        sns.histplot(stress_trace_data, kde=True, color="purple", bins=15)
        plt.title("Distribution of Stress Tensor Trace")
        plt.xlabel("Stress Tensor Trace (atomic units)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_VIS_DIR, "stress_trace_distribution.png"))
        plt.close()
        print("✅ 生成：stress_trace_distribution.png")

    # ------------- 可视化7：最大受力与总能量的散点图 -------------
    plt.figure()
    sns.scatterplot(x=df["max_force_magnitude"], y=df["total_energy_eV"], alpha=0.7, color="orange")
    plt.title("Max Ion Force vs Total Energy")
    plt.xlabel("Max Force Magnitude (atomic units)")
    plt.ylabel("Total Energy (eV)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, "max_force_energy_scatter.png"))
    plt.close()
    print("✅ 生成：max_force_energy_scatter.png")

    print("\n========== 可视化完成 ==========")
    print(f"📁 所有可视化文件已保存至：{OUTPUT_VIS_DIR}")


# ======================== 5. 主函数 ========================
def main():
    """程序主入口"""
    # 步骤1：批量处理所有目录，生成新CIF并收集统计数据
    stats_data = batch_process_all_dirs()

    # 步骤2：生成多维度可视化
    generate_visualizations(stats_data)

    print("\n🎉 程序执行完成！")
    print(f"📁 新CIF文件路径：{OUTPUT_CIF_DIR}")
    print(f"📁 可视化文件路径：{OUTPUT_VIS_DIR}")


if __name__ == "__main__":
    main()