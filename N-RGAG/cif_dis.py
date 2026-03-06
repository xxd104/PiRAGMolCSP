import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple

# ===================== 核心修改：增大全局字体配置 =====================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100  # 默认分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存图片的高分辨率

# 全局字体大小调整（核心修改）
plt.rcParams['font.size'] = 14  # 全局默认字体大小
plt.rcParams['axes.labelsize'] = 16  # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 18  # 图表标题字体大小
plt.rcParams['xtick.labelsize'] = 14  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 14  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 14  # 图例字体大小
plt.rcParams['legend.title_fontsize'] = 16  # 图例标题字体大小


def calculate_cell_volume(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> float:
    """
    计算晶胞体积（角度单位：度）
    公式: V = abc * sqrt(1 - cos²α - cos²β - cos²γ + 2cosαcosβcosγ)
    """
    # 角度转弧度
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # 计算体积
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)

    term = 1 - cos_alpha ** 2 - cos_beta ** 2 - cos_gamma ** 2 + 2 * cos_alpha * cos_beta * cos_gamma
    term = max(term, 0)  # 处理数值精度问题导致的负数
    volume = a * b * c * np.sqrt(term)
    return volume


def parse_cif_file(file_path: str) -> Dict:
    """
    解析单个CIF文件，提取关键信息
    返回：包含解析数据的字典（解析失败时返回空字段）
    """
    parsed_data = {
        'filename': os.path.basename(file_path),
        'elements': [],
        'element_counts': {},
        'atom_coords': {},  # 存储元素的分数坐标
        'total_energy_hartree': None,
        'total_energy_ev': None,
        'avg_ion_force': None,
        'max_ion_force': None,
        'stress_tensor': None,
        'stress_trace': None,
        'cell_volume': None,
        'cell_params': {}
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. 解析晶胞参数 (a, b, c, alpha, beta, gamma)
        cell_params_patterns = {
            'a': r'_cell_length_a\s+([\d\.e+-]+)',
            'b': r'_cell_length_b\s+([\d\.e+-]+)',
            'c': r'_cell_length_c\s+([\d\.e+-]+)',
            'alpha': r'_cell_angle_alpha\s+([\d\.e+-]+)',
            'beta': r'_cell_angle_beta\s+([\d\.e+-]+)',
            'gamma': r'_cell_angle_gamma\s+([\d\.e+-]+)'
        }

        for param, pattern in cell_params_patterns.items():
            match = re.search(pattern, content)
            if match:
                parsed_data['cell_params'][param] = float(match.group(1))

        # 计算晶胞体积（参数齐全时）
        if all(p in parsed_data['cell_params'] for p in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']):
            parsed_data['cell_volume'] = calculate_cell_volume(
                parsed_data['cell_params']['a'],
                parsed_data['cell_params']['b'],
                parsed_data['cell_params']['c'],
                parsed_data['cell_params']['alpha'],
                parsed_data['cell_params']['beta'],
                parsed_data['cell_params']['gamma']
            )

        # 2. 解析原子元素、计数和分数坐标
        atom_site_pattern = r'loop_\s+_atom_site_type_symbol\s+_atom_site_label.*?\n(.*?)\n# =========='
        atom_site_match = re.search(atom_site_pattern, content, re.DOTALL)
        if atom_site_match:
            atom_lines = atom_site_match.group(1).strip().split('\n')
            elements = []
            atom_coords = {}
            for line in atom_lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 6:
                        element = parts[0].strip()
                        x = float(parts[3])
                        y = float(parts[4])
                        z = float(parts[5])
                        elements.append(element)
                        # 存储坐标
                        if element not in atom_coords:
                            atom_coords[element] = []
                        atom_coords[element].append([x, y, z])

            parsed_data['elements'] = elements
            parsed_data['atom_coords'] = atom_coords
            # 统计元素数量
            for elem in elements:
                parsed_data['element_counts'][elem] = parsed_data['element_counts'].get(elem, 0) + 1

        # 3. 解析总能量（Hartree和eV）
        energy_hartree_match = re.search(r'Total Energy \(Hartree\):\s+([\d\.\-e+]+)', content)
        if energy_hartree_match:
            parsed_data['total_energy_hartree'] = float(energy_hartree_match.group(1))

        energy_ev_match = re.search(r'Total Energy \(eV\):\s+([\d\.\-e+]+)', content)
        if energy_ev_match:
            parsed_data['total_energy_ev'] = float(energy_ev_match.group(1))

        # 4. 解析平均和最大离子力
        avg_force_match = re.search(r'Avg Ion Force Magnitude:\s+([\d\.\-e+]+)', content)
        if avg_force_match:
            parsed_data['avg_ion_force'] = float(avg_force_match.group(1))

        max_force_match = re.search(r'Max Ion Force Magnitude:\s+([\d\.\-e+]+)', content)
        if max_force_match:
            parsed_data['max_ion_force'] = float(max_force_match.group(1))

        # 5. 修复：优化应力张量解析逻辑（兼容更多格式，提高匹配成功率）
        # 简化正则，放宽换行/空格限制，适配更多CIF文件格式
        stress_pattern = r'Stress_Row_1:\s*([\d\.\-e+]+)\s+([\d\.\-e+]+)\s+([\d\.\-e+]+).*?' \
                         r'Stress_Row_2:\s*([\d\.\-e+]+)\s+([\d\.\-e+]+)\s+([\d\.\-e+]+).*?' \
                         r'Stress_Row_3:\s*([\d\.\-e+]+)\s+([\d\.\-e+]+)\s+([\d\.\-e+]+)'
        stress_match = re.search(stress_pattern, content, re.DOTALL | re.IGNORECASE)
        if stress_match:
            try:
                stress_vals = [float(val) for val in stress_match.groups()]
                stress_tensor = np.array(stress_vals).reshape(3, 3)
                parsed_data['stress_tensor'] = stress_tensor
                parsed_data['stress_trace'] = np.trace(stress_tensor)  # 计算迹
            except (ValueError, TypeError) as e:
                print(f"解析{file_path}应力张量数值失败: {e}")
                parsed_data['stress_trace'] = None

        return parsed_data

    except Exception as e:
        print(f"解析文件失败 {file_path}: {str(e)}")
        return parsed_data


def batch_parse_cif_files(input_dir: str) -> List[Dict]:
    """
    批量解析指定目录下的所有CIF文件
    返回：解析后的数据列表
    """
    all_parsed_data = []
    # 获取所有.cif文件
    cif_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.cif')]

    if not cif_files:
        print(f"在目录 {input_dir} 中未找到任何CIF文件")
        return all_parsed_data

    print(f"找到 {len(cif_files)} 个CIF文件待处理...")

    for cif_file in cif_files:
        file_path = os.path.join(input_dir, cif_file)
        parsed_data = parse_cif_file(file_path)
        # 仅保留包含核心数据（能量+力）的文件
        if parsed_data['total_energy_ev'] is not None and parsed_data['avg_ion_force'] is not None:
            all_parsed_data.append(parsed_data)
        else:
            print(f"跳过 {cif_file} - 缺少能量/力核心数据")

    print(f"成功解析 {len(all_parsed_data)} 个CIF文件（共 {len(cif_files)} 个）")
    return all_parsed_data


def generate_visualizations(parsed_data_list: List[Dict], output_dir: str):
    """
    生成所有指定的可视化图表并保存到输出目录
    """
    # 创建输出目录（不存在则创建）
    os.makedirs(output_dir, exist_ok=True)

    # 提取绘图所需的数组（过滤None值）
    total_energy_ev = np.array([d['total_energy_ev'] for d in parsed_data_list if d['total_energy_ev'] is not None])
    avg_ion_force = np.array([d['avg_ion_force'] for d in parsed_data_list if d['avg_ion_force'] is not None])
    max_ion_force = np.array([d['max_ion_force'] for d in parsed_data_list if d['max_ion_force'] is not None])
    # 修复：严格过滤stress_trace的None/NaN值（参考你的有效代码逻辑）
    stress_trace = np.array([d['stress_trace'] for d in parsed_data_list if
                             d['stress_trace'] is not None and not np.isnan(d['stress_trace'])])
    cell_volumes = np.array([d['cell_volume'] for d in parsed_data_list if d['cell_volume'] is not None])

    # 统计所有文件的元素总计数
    all_element_counts = {}
    for d in parsed_data_list:
        for elem, count in d['element_counts'].items():
            all_element_counts[elem] = all_element_counts.get(elem, 0) + count

    # ========== 1. 元素分布可视化（饼图） ==========
    plt.figure(figsize=(12, 10))  # 增大画布尺寸适配更大字体
    if all_element_counts:
        labels = list(all_element_counts.keys())
        sizes = list(all_element_counts.values())
        # 调整饼图百分比字体大小（新增参数textprops）
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, explode=[0.05] * len(labels),
                textprops={'fontsize': 14})  # 饼图标签和百分比字体大小
        plt.title('Overall Element Distribution Across All CIF Files', pad=20)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_element_distribution_pie.png'), bbox_inches='tight')
        plt.close()
    else:
        print("无元素数据，跳过元素分布饼图")

    # ========== 2. 元素计数柱状图（修复Seaborn警告） ==========
    plt.figure(figsize=(14, 8))  # 增大画布尺寸适配更大字体
    if all_element_counts:
        elements = list(all_element_counts.keys())
        counts = list(all_element_counts.values())
        # 修复：添加hue参数并设置legend=False，解决palette弃用警告
        sns.barplot(x=elements, y=counts, hue=elements, palette='viridis', legend=False)
        plt.title('Total Count of Each Element Across All CIF Files', pad=20)
        plt.xlabel('Element')
        plt.ylabel('Total Count')
        plt.xticks(rotation=45)
        # 强制设置刻度字体大小（双重保障）
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_element_count_bar.png'), bbox_inches='tight')
        plt.close()
    else:
        print("无元素数据，跳过元素计数柱状图")

    # ========== 3. 最大受力与总能量散点图 ==========
    plt.figure(figsize=(12, 8))  # 增大画布尺寸适配更大字体
    if len(max_ion_force) > 0 and len(total_energy_ev) > 0:
        # 确保数据长度一致
        valid_mask = np.logical_and(~np.isnan(max_ion_force), ~np.isnan(total_energy_ev))
        x = max_ion_force[valid_mask]
        y = total_energy_ev[valid_mask]

        sns.scatterplot(x=x, y=y, alpha=0.7, s=80, color='royalblue', edgecolor='black', linewidth=0.5)
        sns.regplot(x=x, y=y, scatter=False, color='crimson', line_kws={'linestyle': '--', 'linewidth': 1.5})
        plt.title('Max Ion Force vs Total Energy (eV)', pad=20)
        plt.xlabel('Max Ion Force Magnitude (atomic units)')
        plt.ylabel('Total Energy (eV)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_max_force_vs_energy.png'), bbox_inches='tight')
        plt.close()
    else:
        print("数据不足，跳过最大受力-总能量散点图")

    # ========== 4. 应力张量迹的分布直方图（修复：参考你的有效代码逻辑） ==========
    plt.figure(figsize=(12, 8))  # 增大画布尺寸适配更大字体
    # 修复：严格检查非空且非NaN，参考你的df.dropna()逻辑
    if len(stress_trace) > 0 and not np.isnan(stress_trace).all():
        # 过滤NaN值（双重保险）
        valid_stress = stress_trace[~np.isnan(stress_trace)]
        sns.histplot(valid_stress, bins=15, kde=True, color='purple', alpha=0.7, edgecolor='black')
        plt.title('Distribution of Stress Tensor Trace', pad=20)
        plt.xlabel('Stress Tensor Trace (atomic units)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_stress_trace_distribution.png'), bbox_inches='tight')
        plt.close()
        print(f"成功生成应力张量迹分布直方图，共处理 {len(valid_stress)} 个有效数据点")
    else:
        print("无有效应力张量迹数据，跳过应力张量迹分布直方图")

    # ========== 5. 总能量与平均受力相关性散点图 ==========
    plt.figure(figsize=(12, 8))  # 增大画布尺寸适配更大字体
    if len(avg_ion_force) > 0 and len(total_energy_ev) > 0:
        valid_mask = np.logical_and(~np.isnan(avg_ion_force), ~np.isnan(total_energy_ev))
        x = avg_ion_force[valid_mask]
        y = total_energy_ev[valid_mask]

        sns.scatterplot(x=x, y=y, alpha=0.7, s=80, color='purple', edgecolor='black', linewidth=0.5)
        sns.regplot(x=x, y=y, scatter=False, color='orange', line_kws={'linestyle': '--', 'linewidth': 1.5})
        plt.title('Average Ion Force vs Total Energy (eV)', pad=20)
        plt.xlabel('Average Ion Force Magnitude (atomic units)')
        plt.ylabel('Total Energy (eV)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '5_avg_force_vs_energy.png'), bbox_inches='tight')
        plt.close()
    else:
        print("数据不足，跳过平均受力-总能量散点图")

    # ========== 6. 平均受力大小分布直方图 ==========
    plt.figure(figsize=(12, 8))  # 增大画布尺寸适配更大字体
    if len(avg_ion_force) > 0:
        sns.histplot(avg_ion_force, bins=25, kde=True, color='orange', alpha=0.7, edgecolor='black')
        plt.title('Distribution of Average Ion Force Magnitude', pad=20)
        plt.xlabel('Average Ion Force (atomic units)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '6_avg_force_distribution.png'), bbox_inches='tight')
        plt.close()
    else:
        print("无平均受力数据，跳过平均受力分布直方图")

    # ========== 7. 总能量分布直方图 ==========
    plt.figure(figsize=(12, 8))  # 增大画布尺寸适配更大字体
    if len(total_energy_ev) > 0:
        sns.histplot(total_energy_ev, bins=25, kde=True, color='crimson', alpha=0.7, edgecolor='black')
        plt.title('Distribution of Total Energy (eV)', pad=20)
        plt.xlabel('Total Energy (eV)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '7_energy_distribution.png'), bbox_inches='tight')
        plt.close()
    else:
        print("无能量数据，跳过总能量分布直方图")

    # ========== 8. 晶胞体积分布直方图（额外补充） ==========
    plt.figure(figsize=(12, 8))  # 增大画布尺寸适配更大字体
    if len(cell_volumes) > 0:
        sns.histplot(cell_volumes, bins=25, kde=True, color='teal', alpha=0.7, edgecolor='black')
        plt.title('Distribution of Unit Cell Volume', pad=20)
        plt.xlabel('Cell Volume (Å³)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '8_cell_volume_distribution.png'), bbox_inches='tight')
        plt.close()
    else:
        print("无晶胞参数，跳过晶胞体积分布直方图")

    # ========== 9. 3D原子空间分布（第一个有效文件示例） ==========
    first_valid_file = next((d for d in parsed_data_list if d['atom_coords']), None)
    if first_valid_file:
        fig = plt.figure(figsize=(14, 12))  # 增大3D图画布尺寸
        ax = fig.add_subplot(111, projection='3d')

        # 元素颜色映射（常见元素）
        elem_colors = {'H': 'gray', 'C': 'black', 'N': 'blue', 'O': 'red', 'S': 'yellow',
                       'P': 'orange', 'F': 'green', 'Cl': 'green', 'Br': 'brown', 'I': 'purple'}

        # 绘制各元素原子
        for elem, coords in first_valid_file['atom_coords'].items():
            coords = np.array(coords)
            color = elem_colors.get(elem, 'magenta')  # 未知元素用品红色
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       label=elem, color=color, s=200, alpha=0.8, edgecolor='white', linewidth=0.5)

        ax.set_title(f'3D Atomic Spatial Distribution - {first_valid_file["filename"]}', pad=20)
        ax.set_xlabel('Fractional X', labelpad=15)  # 增加标签间距避免重叠
        ax.set_ylabel('Fractional Y', labelpad=15)
        ax.set_zlabel('Fractional Z', labelpad=15)
        ax.legend(loc='best')
        # 增大3D图刻度字体
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                 f'9_3d_atomic_distribution_{first_valid_file["filename"].replace(".cif", "")}.png'),
                    bbox_inches='tight')
        plt.close()
    else:
        print("无原子坐标数据，跳过3D原子分布可视化")

    print(f"\n所有可视化图表已保存至: {output_dir}")


if __name__ == "__main__":
    # 定义输入输出目录
    INPUT_DIR = '/home/nyx/N-RGAG/raw_cifs'
    OUTPUT_DIR = '/home/nyx/N-RGAG/raw_vis'

    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"错误：输入目录 {INPUT_DIR} 不存在！")
        exit(1)

    # 批量解析CIF文件
    parsed_data = batch_parse_cif_files(INPUT_DIR)

    # 生成可视化（有数据时）
    if parsed_data:
        generate_visualizations(parsed_data, OUTPUT_DIR)
    else:
        print("无有效解析数据，无法生成可视化")