import os
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

# ===================== 路径配置（可根据需要调整） =====================
# 标准CIF目录（存放参考结构）
REF_CIF_DIR = "/home/nyx/N-RGAG/old_cif"
# 待比较CIF目录
NEW_CIF_DIR = "/home/nyx/N-RGAG/new_cif"
# 结果保存目录
RESULT_DIR = "/home/nyx/N-RGAG/cal"
# 结果文件名
RESULT_FILENAME = "cif_comparison_best_result.csv"  # 文件名略作修改，体现只保存最优结果


# ===================== 工具函数定义 =====================
def get_reference_structure(ref_dir):
    """
    从参考目录读取标准CIF文件（默认读取第一个.cif文件）
    :param ref_dir: 参考CIF目录路径
    :return: 标准晶体结构（Structure对象）
    """
    # 遍历目录找所有.cif文件
    ref_cif_files = [f for f in os.listdir(ref_dir) if f.endswith(".cif")]
    if not ref_cif_files:
        raise FileNotFoundError(f"参考目录 {ref_dir} 下未找到任何.cif文件！")

    # 取第一个CIF作为标准（如果有多个，可手动指定文件名）
    ref_cif_path = os.path.join(ref_dir, ref_cif_files[0])
    print(f"使用标准CIF文件：{ref_cif_path}")

    # 读取CIF结构
    try:
        ref_structure = Structure.from_file(ref_cif_path)
        return ref_structure
    except Exception as e:
        raise RuntimeError(f"读取标准CIF失败：{e}")


def calculate_cell_similarity(ref_cell, new_cell):
    """
    计算晶胞参数相似度百分比
    晶胞参数：a, b, c (Å)；α, β, γ (°)
    相似度公式：100 - |(新值 - 参考值)/参考值| * 100 （避免除零错误）
    :param ref_cell: 标准晶胞参数（dict，包含a,b,c,alpha,beta,gamma）
    :param new_cell: 新晶胞参数（同ref_cell）
    :return: 各参数相似度 + 平均相似度（dict）
    """
    similarity = {}
    # 遍历所有晶胞参数
    for key in ["a", "b", "c", "alpha", "beta", "gamma"]:
        ref_val = ref_cell[key]
        new_val = new_cell[key]

        # 避免除零错误（角度参数理论上不会为0，长度参数为0则相似度为0）
        if abs(ref_val) < 1e-6:
            sim = 0.0
        else:
            sim = 100 - (abs(new_val - ref_val) / ref_val) * 100

        # 限制相似度范围（避免出现负数）
        similarity[f"{key}_similarity(%)"] = max(0.0, sim)

    # 计算平均相似度
    avg_sim = np.mean(list(similarity.values()))
    similarity["average_similarity(%)"] = avg_sim

    return similarity


def calculate_rmsd(ref_struct, new_struct):
    """
    计算原子位置的RMSD（先对齐结构，保证计算准确）
    :param ref_struct: 标准结构（Structure）
    :param new_struct: 新结构（Structure）
    :return: RMSD值（Å）
    """
    # 初始化结构匹配器（忽略元素种类？False=严格匹配元素，True=只匹配位置）
    matcher = StructureMatcher(
        ltol=0.2,  # 长度容忍度
        stol=0.3,  # 角度容忍度
        angle_tol=5,  # 角度容忍度
        primitive_cell=False  # 使用原胞计算
    )

    try:
        # 对齐结构并计算RMSD
        rmsd = matcher.get_rmsd(ref_struct, new_struct)
        return rmsd
    except Exception as e:
        print(f"RMSD计算失败：{e}")
        return np.nan


# ===================== 主流程 =====================
def main():
    # 1. 创建结果目录（如果不存在）
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_path = os.path.join(RESULT_DIR, RESULT_FILENAME)

    # 2. 读取标准结构
    try:
        ref_struct = get_reference_structure(REF_CIF_DIR)
        # 提取标准晶胞参数
        ref_cell = {
            "a": ref_struct.lattice.a,
            "b": ref_struct.lattice.b,
            "c": ref_struct.lattice.c,
            "alpha": ref_struct.lattice.alpha,
            "beta": ref_struct.lattice.beta,
            "gamma": ref_struct.lattice.gamma
        }
    except Exception as e:
        print(f"读取标准结构失败：{e}")
        return

    # 3. 遍历新CIF目录下的所有CIF文件
    new_cif_files = [f for f in os.listdir(NEW_CIF_DIR) if f.endswith(".cif")]
    if not new_cif_files:
        print(f"待比较目录 {NEW_CIF_DIR} 下未找到任何.cif文件！")
        return

    # 4. 初始化结果列表
    results = []

    # 5. 逐个计算
    for cif_file in new_cif_files:
        cif_path = os.path.join(NEW_CIF_DIR, cif_file)
        print(f"正在处理：{cif_file}")

        # 读取新CIF结构
        try:
            new_struct = Structure.from_file(cif_path)
        except Exception as e:
            print(f"跳过 {cif_file}：读取失败 - {e}")
            continue

        # 提取新晶胞参数
        new_cell = {
            "a": new_struct.lattice.a,
            "b": new_struct.lattice.b,
            "c": new_struct.lattice.c,
            "alpha": new_struct.lattice.alpha,
            "beta": new_struct.lattice.beta,
            "gamma": new_struct.lattice.gamma
        }

        # 计算晶胞相似度
        cell_sim = calculate_cell_similarity(ref_cell, new_cell)

        # 计算RMSD
        rmsd = calculate_rmsd(ref_struct, new_struct)

        # 整理结果
        result = {
            "filename": cif_file,
            "rmsd(Å)": rmsd,
            **cell_sim  # 合并晶胞相似度字典
        }
        results.append(result)

    # 6. 筛选最优结果（核心修改部分）
    if results:
        # 将结果转为DataFrame，方便筛选
        df = pd.DataFrame(results)

        # 步骤1：先按平均相似度降序排序（越高越好）
        # 步骤2：再按RMSD升序排序（越低越好）
        # 步骤3：取排序后的第一条（最优结果）
        df_sorted = df.sort_values(
            by=["average_similarity(%)", "rmsd(Å)"],
            ascending=[False, True]  # 相似度降序，RMSD升序
        )
        best_result = df_sorted.head(1)  # 只保留最优的一条

        # 保存最优结果到CSV
        best_result.to_csv(result_path, index=False, encoding="utf-8")
        print(f"最优结果已保存到：{result_path}")
        print(f"最优匹配的CIF文件：{best_result['filename'].values[0]}")
        print(f"其平均晶胞相似度：{best_result['average_similarity(%)'].values[0]:.2f}%")
        print(f"其原子位置RMSD：{best_result['rmsd(Å)'].values[0]:.4f} Å")
    else:
        print("无有效CIF文件完成计算，未生成结果文件。")


if __name__ == "__main__":
    main()