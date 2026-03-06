#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import time
import glob
import re
from ase.io import read, write
from ase.calculators.vasp import Vasp
import numpy as np

# ========================
# 配置参数 (根据实际情况修改)
# ========================
mid_cifs_dir = "/home/nyx/RG-RAG/mid_cifs"
high_cifs_dir = "/home/nyx/RG-RAG/high_cifs"
vasp_cmd = "mpirun -np 48 vasp_std"  # 使用48核并行
potcar_dir = "/path/to/your/vasp_potentials"  # 设置POTCAR目录路径
scratch_dir = "/scratch/nyx/vasp_run"  # 临时工作目录

# VASP 计算参数 (适用于有机分子)
vasp_settings = {
    'encut': 400,
    'ediff': 1E-6,
    'ediffg': -0.01,
    'ibrion': 2,
    'isif': 3,
    'nsw': 200,
    'lreal': 'Auto',
    'algo': 'Normal',
    'prec': 'Normal',
    'ivdw': 12,  # DFT-D3校正
    'lwave': False,
    'lcharg': False,
    'ismear': 0,
    'sigma': 0.05,
    'kspacing': 0.5,  # 自动K点网格
    'gamma': True,
}

# ========================
# 函数定义
# ========================

def prepare_vasp_inputs(temp_dir, cif_path):
    """准备VASP输入文件"""
    # 转换CIF为POSCAR
    atoms = read(cif_path)
    atoms.write(f"{temp_dir}/POSCAR", format='vasp')
    
    # 生成POTCAR (按POSCAR中的元素顺序)
    with open(f"{temp_dir}/POSCAR", 'r') as f:
        lines = f.readlines()
    elements = lines[5].split()
    
    potcar_path = f"{temp_dir}/POTCAR"
    with open(potcar_path, 'wb') as outfile:
        for elem in elements:
            pot = f"{potcar_dir}/{elem}/POTCAR"
            if not os.path.exists(pot):
                raise FileNotFoundError(f"POTCAR for {elem} not found in {potcar_dir}")
            with open(pot, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
    
    # 生成INCAR
    with open(f"{temp_dir}/INCAR", 'w') as f:
        for key, value in vasp_settings.items():
            f.write(f"{key.upper()} = {value}\n")
    
    # 生成KPOINTS (Gamma中心)
    with open(f"{temp_dir}/KPOINTS", 'w') as f:
        f.write("Automatic mesh\n0\nGamma\n1 1 1\n0 0 0\n")

def run_vasp(temp_dir):
    """运行VASP计算"""
    try:
        result = subprocess.run(
            vasp_cmd.split(),
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=36000  # 10小时超时
        )
        with open(f"{temp_dir}/vasp.out", 'w') as f:
            f.write(result.stdout)
        if result.returncode != 0:
            return False, f"VASP error: {result.stderr}"
        return True, "Success"
    except Exception as e:
        return False, str(e)

def get_final_energy(temp_dir):
    """从OSZICAR获取最终能量"""
    try:
        with open(f"{temp_dir}/OSZICAR", 'r') as f:
            lines = f.readlines()
        last_line = lines[-1]
        if 'E0=' not in last_line:
            return None
        energy = float(last_line.split('E0=')[1].split()[0])
        return energy
    except Exception:
        return None

def process_formula_dir(formula, formula_dir):
    """处理单个分子式目录"""
    print(f"Processing formula: {formula}")
    cif_files = glob.glob(f"{formula_dir}/*.cif")
    if not cif_files:
        print(f"No CIF files found in {formula_dir}")
        return
    
    results = []
    
    for cif_path in cif_files:
        cif_name = os.path.basename(cif_path)
        print(f"  Processing {cif_name}...")
        
        # 创建临时工作目录
        temp_dir = os.path.join(scratch_dir, f"{formula}_{cif_name.replace('.cif', '')}")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # 准备并运行VASP
            prepare_vasp_inputs(temp_dir, cif_path)
            success, msg = run_vasp(temp_dir)
            if not success:
                print(f"    VASP failed: {msg}")
                continue
                
            # 获取能量
            energy = get_final_energy(temp_dir)
            if energy is None:
                print("    Failed to extract energy")
                continue
                
            print(f"    Final energy: {energy:.6f} eV")
            
            # 保存弛豫后的结构
            relaxed_cif = os.path.join(temp_dir, "CONTCAR")
            if os.path.exists(relaxed_cif):
                atoms = read(relaxed_cif)
                results.append({
                    'energy': energy,
                    'source': cif_path,
                    'relaxed_structure': atoms
                })
            else:
                print("    CONTCAR not found")
                
        finally:
            # 清理临时文件 (保留日志)
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # 筛选能量最低的5个结构
    if results:
        sorted_results = sorted(results, key=lambda x: x['energy'])[:5]
        save_dir = os.path.join(high_cifs_dir, formula)
        os.makedirs(save_dir, exist_ok=True)
        
        for i, res in enumerate(sorted_results):
            save_path = os.path.join(save_dir, f"{formula}_{i+1}.cif")
            write(save_path, res['relaxed_structure'])
            print(f"  Saved top structure {i+1} at {save_path}")
    else:
        print(f"  No valid results for {formula}")

# ========================
# 主程序
# ========================

def main():
    # 创建必要的目录
    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(high_cifs_dir, exist_ok=True)
    
    # 遍历所有分子式目录
    for formula in os.listdir(mid_cifs_dir):
        formula_dir = os.path.join(mid_cifs_dir, formula)
        if os.path.isdir(formula_dir):
            process_formula_dir(formula, formula_dir)

if __name__ == "__main__":
    main()