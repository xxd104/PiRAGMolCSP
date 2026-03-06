import os
import math
import numpy as np
from pathlib import Path

# ===================== 配置参数 =====================
ROOT_DIR = "/home/nyx/N-RGAG/zxs_gen_new"
OUTPUT_DIR = "/home/nyx/N-RGAG/raw_cifs"
ELEMENT_ORDER = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']

# ===================== 工具函数 =====================
def parse_geom_gen(file_path):
    """解析geom.out.gen：原子信息、晶胞矢量、元素计数"""
    atoms = []
    cell_vectors = []
    element_counts = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        raise ValueError(f"geom.out.gen文件 {file_path} 为空")

    n_atoms = int(lines[0].split()[0])
    element_symbols = lines[1].split()
    
    for line in lines[2:2+n_atoms]:
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"原子行格式错误: {line} (文件: {file_path})")
        
        elem = element_symbols[int(parts[1])-1]
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        atoms.append((elem, x, y, z))
        element_counts[elem] = element_counts.get(elem, 0) + 1

    for line in lines[-3:]:
        parts = line.split()
        cell_vectors.append([float(parts[0]), float(parts[1]), float(parts[2])])

    return atoms, cell_vectors, element_counts

def calculate_cell_parameters(cell_vectors):
    """从晶胞矢量计算a/b/c/alpha/beta/gamma"""
    a1, a2, a3 = np.array(cell_vectors)
    a, b, c = np.linalg.norm(a1), np.linalg.norm(a2), np.linalg.norm(a3)

    def angle_between(v1, v2):
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0.0
        return np.degrees(np.arccos(max(min(dot/norm, 1.0), -1.0)))

    alpha = angle_between(a2, a3)
    beta = angle_between(a1, a3)
    gamma = angle_between(a1, a2)

    return a, b, c, alpha, beta, gamma

def cartesian_to_fractional(cart_coords, cell_vectors):
    """笛卡尔坐标转分数坐标"""
    cell_matrix = np.array(cell_vectors).T
    try:
        cell_inv = np.linalg.inv(cell_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("晶胞矩阵不可逆，无法转换分数坐标")
    return np.dot(cell_inv, np.array(cart_coords))

def parse_detailed_out(file_path):
    """
    终极解析版：解决所有格式问题
    - 总能量：兼容两种格式
    - 离子受力：优先Total Forces下的数值行
    - 应力张量：先定位Total stress tensor行，强制取其后3行
    """
    total_energy_hartree = None
    total_energy_ev = None
    ion_forces = []
    stress_tensor = []

    # 读取原始行（保留空行，仅去除首尾空格）
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_lines = [line.strip() for line in f.readlines()]

    # ---------------------- 1. 解析总能量 ----------------------
    for line in raw_lines:
        if line.startswith("Total energy:"):
            parts = [p for p in line.split() if p]
            try:
                # 格式1: Total energy: xxx H xxx eV
                if len(parts) >= 6 and parts[3] == 'H' and parts[5] == 'eV':
                    total_energy_hartree = float(parts[2])
                    total_energy_ev = float(parts[4])
                # 格式2: Total energy: xxx Hartree (xxx eV)
                elif len(parts) >= 5 and parts[3] == 'Hartree':
                    total_energy_hartree = float(parts[2])
                    total_energy_ev = float(parts[4].strip('()').split()[0])
            except (ValueError, IndexError):
                continue

    # ---------------------- 2. 解析离子受力 ----------------------
    force_start_idx = -1
    # 找到Total Forces的行号
    for i, line in enumerate(raw_lines):
        if line.startswith("Total Forces"):
            force_start_idx = i + 1
            break
    # 提取受力行
    if force_start_idx != -1:
        for line in raw_lines[force_start_idx:]:
            parts = [p for p in line.split() if p]
            if len(parts) >= 4:
                try:
                    # 验证：第一个是序号，后三个是受力
                    int(parts[0])
                    fx, fy, fz = float(parts[1]), float(parts[2]), float(parts[3])
                    ion_forces.append([fx, fy, fz])
                except ValueError:
                    break  # 非受力行，停止解析
    # 兼容旧格式：Force_xxx行
    if not ion_forces:
        for line in raw_lines:
            if line.startswith("Force_"):
                parts = line.split(':')
                if len(parts) >= 2:
                    vals = [v for v in parts[1].split() if v]
                    if len(vals) >= 3:
                        try:
                            ion_forces.append([float(vals[0]), float(vals[1]), float(vals[2])])
                        except ValueError:
                            continue

    # ---------------------- 3. 解析应力张量（核心修复） ----------------------
    stress_start_idx = -1
    # 找到Total stress tensor的行号
    for i, line in enumerate(raw_lines):
        if line.startswith("Total stress tensor"):
            stress_start_idx = i + 1
            break
    # 强制提取其后3行（跳过空行）
    if stress_start_idx != -1:
        current_idx = stress_start_idx
        stress_count = 0
        while stress_count < 3 and current_idx < len(raw_lines):
            line = raw_lines[current_idx]
            parts = [p for p in line.split() if p]
            if len(parts) >= 3:
                try:
                    sx, sy, sz = float(parts[0]), float(parts[1]), float(parts[2])
                    stress_tensor.append([sx, sy, sz])
                    stress_count += 1
                except ValueError:
                    pass
            current_idx += 1

    # ---------------------- 4. 数据校验 ----------------------
    if total_energy_hartree is None or total_energy_ev is None:
        raise ValueError(f"未找到总能量信息 (文件: {file_path})")
    if not ion_forces:
        raise ValueError(f"未找到离子受力信息 (文件: {file_path})")
    if len(stress_tensor) != 3:
        raise ValueError(f"应力张量不完整（需3行，找到{len(stress_tensor)}行）(文件: {file_path})")

    return total_energy_hartree, total_energy_ev, ion_forces, stress_tensor

def generate_cif_content(atoms, cell_params, element_counts, energy_hartree, energy_ev, ion_forces, stress_tensor, cell_vectors):
    """生成CIF文件内容"""
    a, b, c, alpha, beta, gamma = cell_params

    # 生成化学式
    sorted_elems = sorted(element_counts.keys(), key=lambda x: ELEMENT_ORDER.index(x) if x in ELEMENT_ORDER else 99)
    formula_structural = ''.join([f"{e}{c}" for e, c in element_counts.items()])
    formula_sum = f'"{" ".join([f"{e} {c}" for e, c in element_counts.items()])}"'

    # 构建CIF内容
    cif = [
        "data_image0",
        f"_chemical_formula_structural       {formula_structural}",
        f"_chemical_formula_sum              {formula_sum}",
        f"_cell_length_a       {a}",
        f"_cell_length_b       {b}",
        f"_cell_length_c       {c}",
        f"_cell_angle_alpha    {alpha}",
        f"_cell_angle_beta     {beta}",
        f"_cell_angle_gamma    {gamma}",
        "",
        '_space_group_name_H-M_alt    "P 1"',
        '_space_group_IT_number       1',
        "",
        "loop_",
        "  _space_group_symop_operation_xyz",
        "  'x, y, z'",
        "",
        "loop_",
        "  _atom_site_type_symbol",
        "  _atom_site_label",
        "  _atom_site_symmetry_multiplicity",
        "  _atom_site_fract_x",
        "  _atom_site_fract_y",
        "  _atom_site_fract_z",
        "  _atom_site_occupancy"
    ]

    # 原子信息
    elem_counter = {}
    for elem, x, y, z in atoms:
        elem_counter[elem] = elem_counter.get(elem, 0) + 1
        label = f"{elem}{elem_counter[elem]}"
        frac_x, frac_y, frac_z = cartesian_to_fractional((x, y, z), cell_vectors)
        cif.append(f"  {elem}   {label}        1.0  {frac_x}  {frac_y}  {frac_z}  1.0000")

    # DFTB+结果
    cif.extend([
        "",
        "# ========== DFTB+ Calculation Results (Auto-Generated) ==========",
        f"# Total Energy (Hartree): {energy_hartree}",
        f"# Total Energy (eV): {energy_ev}",
        "# Ion Forces (x, y, z) [atomic units]"
    ])
    # 离子受力
    for i, (fx, fy, fz) in enumerate(ion_forces, 1):
        cif.append(f"# Force_{i:03d}: {fx} {fy} {fz}")
    # 应力张量
    cif.append("# Stress Tensor (3x3) [atomic units]")
    for i, (sx, sy, sz) in enumerate(stress_tensor, 1):
        cif.append(f"# Stress_Row_{i}: {sx} {sy} {sz}")
    # 受力幅值
    force_mags = [np.linalg.norm(f) for f in ion_forces]
    cif.extend([
        f"# Avg Ion Force Magnitude: {np.mean(force_mags)}",
        f"# Max Ion Force Magnitude: {np.max(force_mags)}",
        "# Element Composition: "
    ])

    return '\n'.join(cif)

# ===================== 主程序 =====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")
    print("开始遍历并处理目录...\n")

    for root, dirs, files in os.walk(ROOT_DIR):
        if "detailed.out" in files and "geom.out.gen" in files:
            dir_name = os.path.basename(root)
            geom_path = os.path.join(root, "geom.out.gen")
            detailed_path = os.path.join(root, "detailed.out")
            cif_path = os.path.join(OUTPUT_DIR, f"{dir_name}.cif")

            try:
                print(f"处理目录: {root}")
                # 解析文件
                atoms, cell_vectors, elem_counts = parse_geom_gen(geom_path)
                cell_params = calculate_cell_parameters(cell_vectors)
                energy_h, energy_ev, forces, stress = parse_detailed_out(detailed_path)
                # 生成并保存CIF
                cif_content = generate_cif_content(atoms, cell_params, elem_counts, energy_h, energy_ev, forces, stress, cell_vectors)
                with open(cif_path, 'w', encoding='utf-8') as f:
                    f.write(cif_content)
                print(f"✅ 成功生成: {cif_path}\n")

            except Exception as e:
                print(f"❌ 处理失败: {str(e)}\n")
                continue

    print("处理完成！")

if __name__ == "__main__":
    main()