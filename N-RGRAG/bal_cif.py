import os
import re
import numpy as np
from math import isclose
from pathlib import Path
from collections import defaultdict
import spglib


class CIFCorrector:
    def __init__(self, symprec=1e-5, angle_tolerance=-1.0):
        """
        初始化CIF修正器

        参数:
        symprec: 对称性容差
        angle_tolerance: 角度容差（-1表示自动）
        """
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

        # 原子类型到原子序数的映射
        self.element_to_z = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
            'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
            'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
            'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
            'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
            'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
            'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
            'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111,
            'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117,
            'Og': 118
        }

        # 原子序数到原子符号的映射
        self.z_to_element = {v: k for k, v in self.element_to_z.items()}

    def parse_cif(self, filepath):
        """解析CIF文件"""
        data = {
            'cell': {},
            'symmetry': {},
            'atoms': [],
            'formula': {},
            'comments': [],
            'other_data': {},
            'loop_blocks': []
        }

        with open(filepath, 'r') as f:
            content = f.read()

        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 跳过空行
            if not line:
                i += 1
                continue

            # 处理注释
            if line.startswith('#'):
                data['comments'].append(line)
                i += 1
                continue

            # 处理循环数据
            if line.startswith('loop_'):
                loop_data = {
                    'headers': [],
                    'data': [],
                    'raw_lines': [line]
                }
                i += 1

                # 收集循环头
                while i < len(lines) and lines[i].strip().startswith('_'):
                    header = lines[i].strip()
                    loop_data['headers'].append(header)
                    loop_data['raw_lines'].append(lines[i])
                    i += 1

                # 收集循环数据
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('_'):
                    if not lines[i].strip().startswith('#'):
                        row = lines[i].strip().split()
                        if row:  # 跳过空行
                            loop_data['data'].append(row)
                    loop_data['raw_lines'].append(lines[i])
                    i += 1

                data['loop_blocks'].append(loop_data)

                # 特别处理原子位置循环
                if '_atom_site_label' in loop_data['headers']:
                    for j, header in enumerate(loop_data['headers']):
                        if '_atom_site_fract_x' in header:
                            x_idx = j
                        elif '_atom_site_fract_y' in header:
                            y_idx = j
                        elif '_atom_site_fract_z' in header:
                            z_idx = j
                        elif '_atom_site_type_symbol' in header:
                            type_idx = j
                        elif '_atom_site_label' in header:
                            label_idx = j
                        elif '_atom_site_occupancy' in header:
                            occ_idx = j
                        else:
                            occ_idx = -1

                    for row in loop_data['data']:
                        if len(row) > max(x_idx, y_idx, z_idx, type_idx, label_idx):
                            atom = {
                                'label': row[label_idx],
                                'type': row[type_idx],
                                'x': float(row[x_idx]),
                                'y': float(row[y_idx]),
                                'z': float(row[z_idx]),
                                'occupancy': float(row[occ_idx]) if occ_idx != -1 and occ_idx < len(row) else 1.0
                            }
                            data['atoms'].append(atom)
                continue

            # 处理单行数据
            elif line.startswith('_'):
                # 处理多行值的情况
                key = line.split()[0] if ' ' in line else line

                # 查找值的开始位置
                value_start = line[len(key):].strip()

                # 如果值以引号开始，可能需要跨越多行
                if value_start.startswith("'"):
                    if value_start.count("'") % 2 == 1:  # 奇数个引号，需要继续读取
                        value_lines = [value_start]
                        i += 1
                        while i < len(lines) and not (
                                lines[i].strip().startswith('_') or lines[i].strip().startswith('loop_')):
                            value_lines.append(lines[i])
                            i += 1
                        value = '\n'.join(value_lines)
                    else:
                        value = value_start
                        i += 1
                elif value_start.startswith('"'):
                    if value_start.count('"') % 2 == 1:
                        value_lines = [value_start]
                        i += 1
                        while i < len(lines) and not (
                                lines[i].strip().startswith('_') or lines[i].strip().startswith('loop_')):
                            value_lines.append(lines[i])
                            i += 1
                        value = '\n'.join(value_lines)
                    else:
                        value = value_start
                        i += 1
                else:
                    if value_start:
                        value = value_start.split('#')[0].strip()  # 移除注释
                        i += 1
                    else:
                        # 值在下一行
                        i += 1
                        value_lines = []
                        while i < len(lines) and not (
                                lines[i].strip().startswith('_') or lines[i].strip().startswith('loop_') or lines[
                            i].strip().startswith('#')):
                            value_lines.append(lines[i].strip())
                            i += 1
                        value = ' '.join(value_lines)

                # 清理值
                if isinstance(value, str):
                    value = value.strip()
                    if (value.startswith("'") and value.endswith("'")) or (
                            value.startswith('"') and value.endswith('"')):
                        value = value[1:-1]

                # 晶胞参数
                if '_cell_length' in key:
                    data['cell'][key.split('_')[-1]] = float(value)
                elif '_cell_angle' in key:
                    data['cell'][key.split('_')[-1]] = float(value)
                elif key == '_cell_formula_units_Z':
                    try:
                        data['cell']['Z'] = int(float(value))
                    except:
                        data['cell']['Z'] = 1
                # 对称性
                elif '_symmetry' in key:
                    if 'space_group_name' in key:
                        data['symmetry']['space_group'] = value
                    elif 'Int_Tables_number' in key:
                        try:
                            data['symmetry']['space_group_number'] = int(value)
                        except:
                            data['symmetry']['space_group_number'] = 1
                # 化学式
                elif '_chemical_formula' in key:
                    if 'structural' in key:
                        data['formula']['structural'] = value
                    elif 'sum' in key:
                        data['formula']['sum'] = value
                else:
                    data['other_data'][key] = value
            else:
                i += 1

        return data

    def analyze_with_spglib(self, data):
        """使用spglib分析对称性"""
        if not data['atoms']:
            return None

        # 获取晶胞参数
        a = data['cell'].get('a', 1.0)
        b = data['cell'].get('b', 1.0)
        c = data['cell'].get('c', 1.0)
        alpha = np.radians(data['cell'].get('alpha', 90.0))
        beta = np.radians(data['cell'].get('beta', 90.0))
        gamma = np.radians(data['cell'].get('gamma', 90.0))

        # 计算晶格向量
        # 使用标准晶体学约定
        lattice = np.zeros((3, 3))
        lattice[0, 0] = a
        lattice[1, 0] = b * np.cos(gamma)
        lattice[1, 1] = b * np.sin(gamma)
        lattice[2, 0] = c * np.cos(beta)
        lattice[2, 1] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        lattice[2, 2] = np.sqrt(c ** 2 - lattice[2, 0] ** 2 - lattice[2, 1] ** 2)

        # 准备原子位置和类型
        positions = []
        numbers = []

        for atom in data['atoms']:
            positions.append([atom['x'], atom['y'], atom['z']])
            element = atom['type']
            if element in self.element_to_z:
                numbers.append(self.element_to_z[element])
            else:
                # 尝试只取第一个字符（对于未知元素）
                element_base = element[0] if element else 'X'
                if element_base in self.element_to_z:
                    numbers.append(self.element_to_z[element_base])
                else:
                    numbers.append(0)  # 未知元素

        # 使用spglib分析对称性
        cell = (lattice, positions, numbers)

        # 首先检查是否是有效的晶体结构
        dataset = spglib.get_symmetry_dataset(cell, symprec=self.symprec,
                                              angle_tolerance=self.angle_tolerance)

        if dataset is None:
            return None

        # 使用标准化函数获得标准晶胞
        standardized_cell = spglib.standardize_cell(cell, to_primitive=False,
                                                    no_idealize=False,
                                                    symprec=self.symprec)

        if standardized_cell is None:
            return dataset

        # 分析标准化晶胞的对称性
        std_dataset = spglib.get_symmetry_dataset(standardized_cell,
                                                  symprec=self.symprec,
                                                  angle_tolerance=self.angle_tolerance)

        if std_dataset is None:
            return dataset

        # 从标准化晶胞中提取信息
        std_lattice, std_positions, std_numbers = standardized_cell

        # 计算标准化晶胞参数
        std_a = np.linalg.norm(std_lattice[0])
        std_b = np.linalg.norm(std_lattice[1])
        std_c = np.linalg.norm(std_lattice[2])

        std_alpha = np.degrees(np.arccos(np.dot(std_lattice[1], std_lattice[2]) / (std_b * std_c)))
        std_beta = np.degrees(np.arccos(np.dot(std_lattice[0], std_lattice[2]) / (std_a * std_c)))
        std_gamma = np.degrees(np.arccos(np.dot(std_lattice[0], std_lattice[1]) / (std_a * std_b)))

        # 准备返回结果
        result = {
            'dataset': std_dataset,
            'std_cell': standardized_cell,
            'std_cell_params': {
                'a': std_a,
                'b': std_b,
                'c': std_c,
                'alpha': std_alpha,
                'beta': std_beta,
                'gamma': std_gamma
            },
            'std_atoms': []
        }

        # 转换标准化晶胞中的原子
        for i, (pos, num) in enumerate(zip(std_positions, std_numbers)):
            element = self.z_to_element.get(num, f"X{num}")
            result['std_atoms'].append({
                'label': f"{element}{i + 1}",
                'type': element,
                'x': pos[0],
                'y': pos[1],
                'z': pos[2],
                'occupancy': 1.0
            })

        return result

    def check_atom_labels(self, atoms):
        """检查原子标签连续性并修正"""
        # 按元素分组
        element_counts = defaultdict(int)

        # 首先找到所有元素类型
        for atom in atoms:
            element = atom['type']
            element_counts[element] += 1

        # 重新编号
        new_atoms = []
        current_counts = defaultdict(int)

        for atom in atoms:
            element = atom['type']
            current_counts[element] += 1
            new_label = f"{element}{current_counts[element]}"

            new_atom = atom.copy()
            new_atom['label'] = new_label
            new_atoms.append(new_atom)

        return new_atoms

    def calculate_z_value(self, atoms, formula_sum):
        """计算正确的Z值"""
        if not formula_sum:
            return 1

        # 从化学式中提取原子数
        formula_parts = re.findall(r'([A-Z][a-z]*)(\d*)', formula_sum)
        formula_atoms = {}

        for element, count in formula_parts:
            count = int(count) if count else 1
            if element in formula_atoms:
                formula_atoms[element] += count
            else:
                formula_atoms[element] = count

        # 统计晶胞中的原子数
        cell_atoms = defaultdict(int)
        for atom in atoms:
            element = atom['type']
            # 处理同位素标记（如D代替H）
            if element == 'D':
                element = 'H'  # 氘视为氢
            elif len(element) > 1 and element[0].isalpha() and element[1:].isdigit():
                # 可能是同位素标记，如C13
                base_element = ''.join([c for c in element if not c.isdigit()])
                if base_element:
                    element = base_element

            cell_atoms[element] += 1

        # 计算Z值（每个化学式单元对应的分子数）
        z_values = []
        for element, count in formula_atoms.items():
            if element in cell_atoms:
                if count > 0:
                    z_values.append(cell_atoms[element] / count)
            else:
                # 如果公式中有原子但晶胞中没有，返回1
                return 1

        # 取平均值并四舍五入到最近的整数
        if z_values:
            avg_z = sum(z_values) / len(z_values)
            return int(round(avg_z))
        return 1

    def generate_symmetry_operations(self, dataset):
        """从spglib数据集生成对称操作"""
        operations = []

        if dataset is None or 'rotations' not in dataset or 'translations' not in dataset:
            # 默认恒等操作
            operations.append('x, y, z')
            return operations

        for rot, trans in zip(dataset['rotations'], dataset['translations']):
            # 转换旋转矩阵和平移向量为字符串表示
            op_parts = []

            for i in range(3):
                term_parts = []

                # x分量
                if rot[i, 0] == 1:
                    term_parts.append('x')
                elif rot[i, 0] == -1:
                    term_parts.append('-x')
                elif rot[i, 0] != 0:
                    term_parts.append(f"{int(rot[i, 0])}*x")

                # y分量
                if rot[i, 1] == 1:
                    if term_parts:
                        term_parts.append('+y')
                    else:
                        term_parts.append('y')
                elif rot[i, 1] == -1:
                    term_parts.append('-y')
                elif rot[i, 1] != 0:
                    if rot[i, 1] > 0 and term_parts:
                        term_parts.append(f"+{int(rot[i, 1])}*y")
                    else:
                        term_parts.append(f"{int(rot[i, 1])}*y")

                # z分量
                if rot[i, 2] == 1:
                    if term_parts:
                        term_parts.append('+z')
                    else:
                        term_parts.append('z')
                elif rot[i, 2] == -1:
                    term_parts.append('-z')
                elif rot[i, 2] != 0:
                    if rot[i, 2] > 0 and term_parts:
                        term_parts.append(f"+{int(rot[i, 2])}*z")
                    else:
                        term_parts.append(f"{int(rot[i, 2])}*z")

                # 平移项
                trans_val = trans[i]
                # 将平移值调整到[0, 1)区间
                trans_val = trans_val % 1
                if abs(trans_val) < 1e-10:
                    trans_val = 0.0
                elif abs(trans_val - 1) < 1e-10:
                    trans_val = 0.0
                elif abs(trans_val + 1) < 1e-10:
                    trans_val = 0.0

                if abs(trans_val) > 1e-10:
                    if trans_val > 0:
                        if term_parts:
                            term_parts.append(f"+{trans_val:.6f}".rstrip('0').rstrip('.'))
                        else:
                            term_parts.append(f"{trans_val:.6f}".rstrip('0').rstrip('.'))
                    else:
                        term_parts.append(f"{trans_val:.6f}".rstrip('0').rstrip('.'))

                term = ''.join(term_parts)
                if i == 0:
                    operation = term
                elif i == 1:
                    operation += f", {term}"
                else:
                    operation += f", {term}"

            operations.append(operation)

        # 去重并排序（恒等操作放在第一位）
        unique_ops = []
        for op in operations:
            if op not in unique_ops:
                unique_ops.append(op)

        # 确保恒等操作在第一位
        if 'x, y, z' in unique_ops:
            unique_ops.remove('x, y, z')

        return ['x, y, z'] + unique_ops

    def write_corrected_cif(self, data, spglib_result, output_path):
        """写入修正后的CIF文件"""
        with open(output_path, 'w') as f:
            # 写入注释
            for comment in data.get('comments', []):
                f.write(f"{comment}\n")

            # 写入创建方法
            f.write("_audit_creation_method 'N-RGAG structure generation (corrected with spglib)'\n")

            # 写入化学式
            if 'formula' in data:
                if 'structural' in data['formula'] and data['formula']['structural']:
                    f.write(f"_chemical_formula_structural '{data['formula']['structural']}'\n")
                if 'sum' in data['formula'] and data['formula']['sum']:
                    f.write(f"_chemical_formula_sum '{data['formula']['sum']}'\n")

            # 写入晶胞参数
            if spglib_result:
                cell_params = spglib_result['std_cell_params']
            else:
                cell_params = data['cell']

            f.write(f"_cell_length_a {cell_params.get('a', 1.0):.6f}\n")
            f.write(f"_cell_length_b {cell_params.get('b', 1.0):.6f}\n")
            f.write(f"_cell_length_c {cell_params.get('c', 1.0):.6f}\n")
            f.write(f"_cell_angle_alpha {cell_params.get('alpha', 90.0):.6f}\n")
            f.write(f"_cell_angle_beta {cell_params.get('beta', 90.0):.6f}\n")
            f.write(f"_cell_angle_gamma {cell_params.get('gamma', 90.0):.6f}\n")

            # 写入Z值
            if 'cell' in data and 'Z' in data['cell']:
                f.write(f"_cell_formula_units_Z {data['cell']['Z']}\n")
            else:
                f.write("_cell_formula_units_Z 1\n")

            # 写入对称性信息
            if spglib_result and spglib_result['dataset']:
                dataset = spglib_result['dataset']
                f.write(f"_symmetry_space_group_name_H-M '{dataset['international']}'\n")
                f.write(f"_symmetry_Int_Tables_number {dataset['number']}\n")
            elif 'symmetry' in data and 'space_group' in data['symmetry']:
                f.write(f"_symmetry_space_group_name_H-M '{data['symmetry']['space_group']}'\n")
                if 'space_group_number' in data['symmetry']:
                    f.write(f"_symmetry_Int_Tables_number {data['symmetry']['space_group_number']}\n")
                else:
                    f.write("_symmetry_Int_Tables_number 1\n")
            else:
                f.write("_symmetry_space_group_name_H-M 'P 1'\n")
                f.write("_symmetry_Int_Tables_number 1\n")

            # 写入对称操作
            f.write("loop_\n")
            f.write("_symmetry_equiv_pos_as_xyz\n")

            if spglib_result and spglib_result['dataset']:
                operations = self.generate_symmetry_operations(spglib_result['dataset'])
                for op in operations:
                    f.write(f"'{op}'\n")
            else:
                f.write("'x, y, z'\n")

            # 写入原子位置
            if spglib_result:
                atoms = spglib_result['std_atoms']
            else:
                atoms = data['atoms']

            if atoms:
                f.write("loop_\n")
                f.write("_atom_site_label\n")
                f.write("_atom_site_type_symbol\n")
                f.write("_atom_site_fract_x\n")
                f.write("_atom_site_fract_y\n")
                f.write("_atom_site_fract_z\n")
                f.write("_atom_site_occupancy\n")

                for atom in atoms:
                    f.write(f"{atom['label']:8} {atom['type']:4} "
                            f"{atom['x']:.8f} {atom['y']:.8f} {atom['z']:.8f} "
                            f"{atom['occupancy']:.6f}\n")

            # 写入原始循环块（非原子位置部分）
            for loop_block in data.get('loop_blocks', []):
                # 跳过原子位置循环，因为已经处理过了
                if '_atom_site_label' in loop_block['headers']:
                    continue

                f.write("loop_\n")
                for header in loop_block['headers']:
                    f.write(f"{header}\n")

                for row in loop_block['data']:
                    f.write(" ".join(row) + "\n")

            # 写入其他数据
            for key, value in data.get('other_data', {}).items():
                if isinstance(value, str) and '\n' in value:
                    f.write(f"{key}\n")
                    f.write(f"{value}\n")
                else:
                    f.write(f"{key} '{value}'\n")

    def correct_cif_file(self, input_path, output_path):
        """修正单个CIF文件"""
        print(f"处理文件: {input_path}")

        try:
            # 解析CIF文件
            data = self.parse_cif(input_path)

            # 使用spglib分析对称性
            print("  使用spglib分析对称性...")
            spglib_result = self.analyze_with_spglib(data)

            if spglib_result:
                print(f"  检测到空间群: {spglib_result['dataset']['international']} "
                      f"(#{spglib_result['dataset']['number']})")
                print(f"  标准化晶胞参数: a={spglib_result['std_cell_params']['a']:.4f} Å, "
                      f"b={spglib_result['std_cell_params']['b']:.4f} Å, "
                      f"c={spglib_result['std_cell_params']['c']:.4f} Å")

                # 使用标准化后的原子
                data['atoms'] = spglib_result['std_atoms']
            else:
                print("  spglib分析失败，使用原始数据")

            # 修正原子标签连续性
            if data['atoms']:
                data['atoms'] = self.check_atom_labels(data['atoms'])
                print(f"  原子标签已重新编号，共 {len(data['atoms'])} 个原子")

            # 计算并修正Z值
            if 'formula' in data and 'sum' in data['formula']:
                new_z = self.calculate_z_value(data['atoms'], data['formula']['sum'])
                if 'cell' not in data:
                    data['cell'] = {}
                if data['cell'].get('Z', 1) != new_z:
                    print(f"  修正Z值: {data['cell'].get('Z', 1)} -> {new_z}")
                    data['cell']['Z'] = new_z

            # 写入修正后的文件
            self.write_corrected_cif(data, spglib_result, output_path)
            print(f"  已保存到: {output_path}")
            print("-" * 50)

            return True

        except Exception as e:
            import traceback
            print(f"  处理文件时出错: {str(e)}")
            traceback.print_exc()
            return False


def process_all_cif_files(input_base_dir, output_base_dir):
    """处理所有CIF文件"""
    corrector = CIFCorrector(symprec=1e-5)

    # 确保输出目录存在
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)

    # 遍历输入目录
    input_path = Path(input_base_dir)

    # 统计
    total_files = 0
    processed_files = 0
    failed_files = 0

    # 查找所有CIF文件
    cif_files = list(input_path.rglob("*.cif"))

    if not cif_files:
        print(f"在目录 {input_base_dir} 中未找到CIF文件")
        return

    print(f"找到 {len(cif_files)} 个CIF文件")
    print("=" * 50)

    for cif_file in cif_files:
        total_files += 1

        # 计算相对路径
        try:
            relative_path = cif_file.relative_to(input_path)
        except ValueError:
            # 如果文件不在输入目录下，使用文件名
            relative_path = Path(cif_file.name)

        # 创建输出路径
        output_file = Path(output_base_dir) / relative_path

        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 修正CIF文件
        if corrector.correct_cif_file(str(cif_file), str(output_file)):
            processed_files += 1
        else:
            failed_files += 1

    # 输出统计信息
    print("=" * 50)
    print("处理完成!")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {processed_files}")
    print(f"处理失败: {failed_files}")


def process_single_file():
    """处理单个文件示例"""
    corrector = CIFCorrector(symprec=1e-5)

    input_file = "/home/nyx/N-RGAG/new_cif/CH4/example.cif"
    output_file = "/home/nyx/N-RGAG/bal_cif/CH4/example_corrected.cif"

    # 确保输出目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    if corrector.correct_cif_file(input_file, output_file):
        print("单文件处理成功!")
    else:
        print("单文件处理失败!")


if __name__ == "__main__":
    # 设置输入输出目录
    input_base_dir = "/home/nyx/N-RGRAG/new_cif"
    output_base_dir = "/home/nyx/N-RGRAG/bal_cif"

    # 检查spglib是否可用
    try:
        import spglib

        print("spglib库已成功导入")
    except ImportError:
        print("错误: 未安装spglib库")
        print("请使用以下命令安装: pip install spglib")
        exit(1)

    # 运行批量处理
    print("开始批量处理CIF文件...")
    process_all_cif_files(input_base_dir, output_base_dir)

    # 或者处理单个文件
    # process_single_file()