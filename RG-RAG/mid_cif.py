import os
import glob
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.dftb import DftbPlusInputSet

# 配置路径参数
INPUT_CIF_DIR = "/home/nyx/RG-RAG/generated_cifs"
OUTPUT_CIF_DIR = "/home/nyx/RG-RAG/mid_cifs"
TEMP_WORK_DIR = "/home/nyx/RG-RAG/temp_dftb_work"
DFTBPLUS_EXECUTABLE = "dftb+"  # DFTB+可执行文件名称
PARAM_DIR = "/path/to/dftb/parameters"  # DFTB+参数文件目录

# 确保目录存在
os.makedirs(OUTPUT_CIF_DIR, exist_ok=True)
os.makedirs(TEMP_WORK_DIR, exist_ok=True)

# 遍历所有类别目录
for category_dir in glob.glob(os.path.join(INPUT_CIF_DIR, "*")):
    if not os.path.isdir(category_dir):
        continue

    category_name = os.path.basename(category_dir)
    print(f"\nProcessing Category: {category_name}")

    # 存储当前类别的结果
    category_results = []

    # 遍历当前类别下的所有CIF文件
    for idx, cif_path in enumerate(glob.glob(os.path.join(category_dir, "*.cif"))):
        struct_name = os.path.basename(cif_path).split('.')[0]
        struct_dir = os.path.join(TEMP_WORK_DIR, f"{category_name}_{idx}")
        os.makedirs(struct_dir, exist_ok=True)

        print(f"  Processing {idx + 1}: {struct_name}")

        try:
            # Step 1: 读取CIF结构
            struct = Structure.from_file(cif_path)

            # Step 2: 创建DFTB+输入文件
            input_set = DftbPlusInputSet(
                struct,
                kpts=(1, 1, 1),  # 可以根据需要调整k点
                charge=0,
                max_scf=100,  # 最大SCF步数
                driver_force_tol=1e-3,  # 力收敛容差
                driver_max_steps=1500,  # 最大优化步数
                dftb_hsd_path=os.path.join(struct_dir, "dftb_in.hsd"),
                dftb_in_path=os.path.join(struct_dir, "dftb_in.hsd"),
                dftb_out_path=os.path.join(struct_dir, "dftb.out"),
                gen_path=os.path.join(struct_dir, "structure.gen"),
                skf_path=PARAM_DIR
            )
            input_set.write_input(struct_dir)

            # Step 3: 运行DFTB+弛豫
            cmd = [DFTBPLUS_EXECUTABLE]
            with open(os.path.join(struct_dir, "dftbplus_output.txt"), "w") as f:
                subprocess.run(
                    cmd,
                    cwd=struct_dir,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    check=True
                )

            # Step 4: 提取弛豫后结构和能量
            # 解析能量 (从dftb.out文件中获取最终能量)
            out_file = os.path.join(struct_dir, "dftb.out")
            with open(out_file, "r") as f:
                lines = f.readlines()

            # 查找最终能量值
            final_energy = None
            for line in reversed(lines):
                if "Total energy:" in line:
                    parts = line.split()
                    final_energy = float(parts[2])
                    break

            if final_energy is None:
                raise ValueError("Energy not found in DFTB+ output")

            # 读取弛豫后的结构
            gen_file = os.path.join(struct_dir, "structure.gen")
            relaxed_struct = Structure.from_file(gen_file)

            # 保存结果
            category_results.append((cif_path, relaxed_struct, final_energy))
            print(f"    Success! Energy = {final_energy:.4f} eV")

        except Exception as e:
            print(f"    Failed: {str(e)}")
            continue

    # 筛选当前类别中能量最低的30个结构
    if category_results:
        category_results.sort(key=lambda x: x[2])  # 按能量升序排序
        top_30 = category_results[:30]

        # 为当前类别创建单独的输出目录
        category_output_dir = os.path.join(OUTPUT_CIF_DIR, category_name)
        os.makedirs(category_output_dir, exist_ok=True)

        print(f"\nTop 30 structures for {category_name}:")
        for i, (orig_path, struct, energy) in enumerate(top_30):
            orig_name = os.path.basename(orig_path)
            output_path = os.path.join(category_output_dir, f"top_{i + 1}_{orig_name}")
            struct.to(filename=output_path, fmt="cif")
            print(f"{i + 1}. {orig_name} -> Energy: {energy:.4f} eV")

        print(f"Saved top {len(top_30)} structures to {category_output_dir}")

        # 绘制当前类别的结合能分布图
        plt.figure(figsize=(10, 6))
        energies = [result[2] for result in top_30]
        names = [os.path.basename(result[0]).split('.')[0] for result in top_30]

        plt.bar(range(1, len(top_30) + 1), energies, color='skyblue')
        plt.xticks(range(1, len(top_30) + 1), rotation=45, ha='right', fontsize=8)
        plt.xlabel('Structure Rank')
        plt.ylabel('Binding Energy (eV)')
        plt.title(f'{category_name} - Binding Energy Distribution of Top 30 Structures')
        plt.tight_layout()

        # 保存图表
        chart_path = os.path.join(category_output_dir, f'binding_energy_distribution_{category_name}.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Energy distribution chart saved to: {chart_path}")

        # 关闭图表以释放内存
        plt.close()
    else:
        print(f"No successful relaxations for {category_name}!")

print("\nAll categories processed!")