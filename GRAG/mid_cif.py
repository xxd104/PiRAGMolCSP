import os
import sys
import subprocess
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import numpy as np


# ==================== 配置部分 ====================
class Config:
    """配置文件"""
    # 基本路径
    BASE_DIR = Path("/home/nyx/GRAG")
    DFTB_BIN = BASE_DIR / "/home/nyx/GRAG/dftbplus/dftbplus-23.1.x86_64-linux/bin/dftb+"

    # 目录结构
    INPUT_CIF_DIR = BASE_DIR / "/home/nyx/GRAG/low_cifs/CH4"
    OUTPUT_CIF_DIR = BASE_DIR / "/home/nyx/GRAG/mid_cifs/CH4"
    WORK_DIR = BASE_DIR / "work"
    LOG_DIR = BASE_DIR / "logs"

    # DFTB+参数
    SKF_DIR = BASE_DIR / "/home/nyx/GRAG/dftbplus/slakos/3ob-3-1"  # 参数集目录
    MAX_STEPS = 500
    FORCE_TOL = 1e-4  # 原子单位
    STRESS_TOL = 1e-5  # 原子单位
    KPOINTS = [2, 2, 2]  # k点网格

    # 计算设置
    OMP_NUM_THREADS = 4
    MKL_NUM_THREADS = 4


# ==================== 系统初始化 ====================
class SystemSetup:
    """系统初始化"""

    @staticmethod
    def check_dependencies():
        """检查依赖"""
        print("检查系统依赖...")

        # 检查DFTB+
        if not Config.DFTB_BIN.exists():
            print(f"错误: 未找到DFTB+可执行文件: {Config.DFTB_BIN}")
            return False

        # 检查参数集
        if not Config.SKF_DIR.exists():
            print(f"警告: 未找到参数集目录: {Config.SKF_DIR}")
            print("请下载参数集: https://dftb.org/parameters/download/3ob")

        # 检查Python包
        required_packages = ['numpy']
        optional_packages = ['ase', 'pymatgen']

        for pkg in required_packages:
            try:
                __import__(pkg)
                print(f"✓ {pkg}")
            except ImportError:
                print(f"✗ 缺少必要包: {pkg}")
                print(f"请安装: pip install {pkg}")
                return False

        for pkg in optional_packages:
            try:
                __import__(pkg)
                print(f"✓ {pkg} (可选)")
            except ImportError:
                print(f"⚠ 缺少可选包: {pkg}，部分功能可能受限")

        return True

    @staticmethod
    def setup_directories():
        """创建目录结构"""
        print("\n创建目录结构...")

        directories = [
            Config.OUTPUT_CIF_DIR,
            Config.WORK_DIR,
            Config.LOG_DIR,
            Config.BASE_DIR / "inputs",
            Config.BASE_DIR / "outputs"
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  创建: {dir_path}")

        print("目录结构创建完成")

    @staticmethod
    def setup_environment():
        """设置环境变量"""
        print("\n设置环境变量...")

        env_vars = {
            'OMP_NUM_THREADS': str(Config.OMP_NUM_THREADS),
            'MKL_NUM_THREADS': str(Config.MKL_NUM_THREADS),
            'DFTB_PREFIX': str(Config.SKF_DIR)
        }

        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"  设置 {key}={value}")

        print("环境变量设置完成")


# ==================== 结构处理 ====================
class StructureProcessor:
    """结构处理工具"""

    @staticmethod
    def cif_to_poscar(cif_file, poscar_file):
        """将CIF转换为POSCAR格式"""
        try:
            # 尝试使用ASE
            try:
                from ase.io import read, write
                atoms = read(cif_file, format='cif')
                write(poscar_file, atoms, format='vasp')
                return True
            except ImportError:
                pass

            # 如果没有ASE，使用简单解析
            return StructureProcessor._simple_cif_to_poscar(cif_file, poscar_file)

        except Exception as e:
            print(f"  转换失败: {e}")
            return False

    @staticmethod
    def _simple_cif_to_poscar(cif_file, poscar_file):
        """简单的CIF到POSCAR转换（基础版本）"""
        try:
            with open(cif_file, 'r') as f:
                lines = f.readlines()

            # 提取晶胞参数
            cell_params = {}
            atom_data = []
            reading_atoms = False

            for line in lines:
                if '_cell_length_a' in line:
                    cell_params['a'] = float(line.split()[-1])
                elif '_cell_length_b' in line:
                    cell_params['b'] = float(line.split()[-1])
                elif '_cell_length_c' in line:
                    cell_params['c'] = float(line.split()[-1])
                elif '_cell_angle_alpha' in line:
                    cell_params['alpha'] = float(line.split()[-1])
                elif '_cell_angle_beta' in line:
                    cell_params['beta'] = float(line.split()[-1])
                elif '_cell_angle_gamma' in line:
                    cell_params['gamma'] = float(line.split()[-1])
                elif '_atom_site_fract_x' in line:
                    reading_atoms = True
                    continue
                elif reading_atoms and line.strip() and not line.startswith('_'):
                    parts = line.split()
                    if len(parts) >= 6:
                        atom_data.append({
                            'symbol': parts[0],
                            'x': float(parts[3]),
                            'y': float(parts[4]),
                            'z': float(parts[5])
                        })

            if not cell_params or not atom_data:
                return False

            # 写入POSCAR
            with open(poscar_file, 'w') as f:
                f.write("Generated from CIF\n")
                f.write("1.0\n")

                # 晶胞向量（简化处理，假设正交）
                f.write(f"{cell_params['a']} 0.0 0.0\n")
                f.write(f"0.0 {cell_params['b']} 0.0\n")
                f.write(f"0.0 0.0 {cell_params['c']}\n")

                # 原子类型和数量
                atom_counts = {}
                for atom in atom_data:
                    atom_counts[atom['symbol']] = atom_counts.get(atom['symbol'], 0) + 1

                f.write(" ".join(atom_counts.keys()) + "\n")
                f.write(" ".join(str(count) for count in atom_counts.values()) + "\n")
                f.write("Direct\n")

                # 原子位置
                for atom in atom_data:
                    f.write(f"{atom['x']} {atom['y']} {atom['z']}\n")

            return True

        except Exception as e:
            print(f"  简单转换失败: {e}")
            return False


# ==================== DFTB+输入生成 ====================
class DFTBInputGenerator:
    """DFTB+输入文件生成器"""

    @staticmethod
    def generate_input(structure_name, work_dir):
        """生成DFTB+输入文件"""
        input_file = work_dir / "dftb_in.hsd"

        template = f"""Geometry = VASPFormat {{
  <<< "POSCAR"
}}

Driver = GeometryOptimization {{
  # 几何优化设置
  MaxSteps = {Config.MAX_STEPS}
  LatticeOpt = Yes  # 同时优化晶胞
  # 收敛标准
  MaxForceComponent = {Config.FORCE_TOL}
  MaxForceRMS = {Config.FORCE_TOL * 2}
  MaxStressComponent = {Config.STRESS_TOL}
  OutputPrefix = "{structure_name}_opt"
}}

Hamiltonian = DFTB {{
  Scc = Yes  # 自洽电荷
  SCCTolerance = 1.0e-7
  MaxSCCIterations = 200
  Mixer = Broyden {{}}

  # 参数集设置
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{Config.SKF_DIR}/"
    Separator = "-"
    Suffix = ".skf"
  }}

  # 角动量设置
  MaxAngularMomentum {{
    H = "s"
    C = "p"
  }}

  # 色散校正 (D3)
  Dispersion = DftD3 {{
    s6 = 1.0
    s8 = 0.722
    a1 = 0.746
    a2 = 4.191
    damping = BeckeJohnson {{
      a1 = 0.746
      a2 = 4.191
    }}
  }}

  # k点设置
  KPointsAndWeights = SupercellFolding {{
    {Config.KPOINTS[0]} 0 0
    0 {Config.KPOINTS[1]} 0
    0 0 {Config.KPOINTS[2]}
    0.5 0.5 0.5
  }}
}}

ParserOptions {{
  ParserVersion = 10
}}

# 输出设置
Analysis {{
  WriteBandOut = No
  WriteEigenvectors = No
  MullikenAnalysis = Yes
  AtomResolvedEnergies = Yes
}}

Options {{
  WriteDetailedXML = Yes
  WriteChargesAsText = Yes
}}
"""

        try:
            with open(input_file, 'w') as f:
                f.write(template)
            print(f"  生成输入文件: {input_file}")
            return True
        except Exception as e:
            print(f"  生成输入文件失败: {e}")
            return False


# ==================== DFTB+计算运行 ====================
class DFTBCalculator:
    """DFTB+计算器"""

    @staticmethod
    def run_calculation(structure_name, work_dir):
        """运行DFTB+计算"""
        print(f"  开始DFTB+计算...")

        # 进入工作目录
        original_dir = os.getcwd()
        os.chdir(work_dir)

        try:
            # 运行DFTB+
            start_time = time.time()

            with open("dftb.out", "w") as outfile:
                result = subprocess.run(
                    [str(Config.DFTB_BIN)],
                    stdout=outfile,
                    stderr=subprocess.STDOUT,
                    text=True
                )

            end_time = time.time()
            elapsed = end_time - start_time

            os.chdir(original_dir)

            if result.returncode == 0:
                print(f"  计算成功完成，耗时: {elapsed:.1f}秒")
                return True, elapsed
            else:
                print(f"  计算失败，返回码: {result.returncode}")
                return False, elapsed

        except Exception as e:
            os.chdir(original_dir)
            print(f"  运行DFTB+时出错: {e}")
            return False, 0


# ==================== 结果处理 ====================
class ResultProcessor:
    """结果处理器"""

    @staticmethod
    def extract_results(calc_dir, structure_name):
        """提取计算结果"""
        print(f"  提取结果...")

        results = {
            'success': False,
            'energy': None,
            'output_cif': None,
            'forces': [],
            'stress': None
        }

        try:
            # 提取能量
            energy = ResultProcessor._extract_energy(calc_dir)
            results['energy'] = energy

            # 转换结构文件
            cif_file = ResultProcessor._convert_to_cif(calc_dir, structure_name)
            if cif_file:
                results['output_cif'] = cif_file
                results['success'] = True

            # 提取力和应力
            results['forces'] = ResultProcessor._extract_forces(calc_dir)
            results['stress'] = ResultProcessor._extract_stress(calc_dir)

        except Exception as e:
            print(f"  提取结果时出错: {e}")

        return results

    @staticmethod
    def _extract_energy(calc_dir):
        """提取总能量"""
        # 尝试从detailed.xml提取
        xml_file = calc_dir / "detailed.xml"
        if xml_file.exists():
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for elem in root.iter('TotalEnergy'):
                    if 'value' in elem.attrib:
                        energy = float(elem.attrib['value'])
                        print(f"    总能量: {energy:.6f} Hartree")
                        return energy
            except Exception as e:
                print(f"    XML解析失败: {e}")

        # 尝试从输出文件提取
        out_file = calc_dir / "dftb.out"
        if out_file.exists():
            try:
                with open(out_file, 'r') as f:
                    for line in f:
                        if 'Total energy' in line and ':' in line:
                            parts = line.split(':')
                            if len(parts) > 1:
                                energy = float(parts[1].strip())
                                print(f"    总能量: {energy:.6f} Hartree")
                                return energy
            except Exception as e:
                print(f"    文本解析失败: {e}")

        return None

    @staticmethod
    def _convert_to_cif(calc_dir, structure_name):
        """将结果转换为CIF格式"""
        # 查找生成的结构文件
        gen_files = list(calc_dir.glob("*_opt.gen")) + list(calc_dir.glob("geo_end.gen"))

        if not gen_files:
            print(f"    未找到结构文件")
            return None

        gen_file = gen_files[0]
        output_cif = Config.OUTPUT_CIF_DIR / f"{structure_name}_relaxed.cif"

        try:
            # 尝试使用ASE转换
            try:
                from ase.io import read, write
                atoms = read(gen_file, format='gen')
                write(output_cif, atoms, format='cif')
                print(f"    生成CIF文件: {output_cif}")
                return output_cif
            except ImportError:
                pass

            # 简单转换
            shutil.copy(gen_file, output_cif.with_suffix('.gen'))
            print(f"    保存GEN文件: {output_cif.with_suffix('.gen')}")
            return output_cif.with_suffix('.gen')

        except Exception as e:
            print(f"    转换结构文件失败: {e}")
            return None

    @staticmethod
    def _extract_forces(calc_dir):
        """提取原子力"""
        forces = []
        # 这里可以添加解析力的代码
        return forces

    @staticmethod
    def _extract_stress(calc_dir):
        """提取应力"""
        # 这里可以添加解析应力的代码
        return None


# ==================== 结构分析 ====================
class StructureAnalyzer:
    """结构分析工具"""

    @staticmethod
    def analyze_bond_lengths(cif_file):
        """分析键长"""
        try:
            from ase.io import read
            atoms = read(cif_file, format='cif')

            # 分离碳和氢原子
            c_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'C']
            h_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'H']

            bond_lengths = []
            for c_idx in c_indices:
                for h_idx in h_indices:
                    dist = atoms.get_distance(c_idx, h_idx, mic=True)
                    if dist < 1.3:  # C-H键阈值
                        bond_lengths.append(dist)

            if bond_lengths:
                stats = {
                    'mean': np.mean(bond_lengths),
                    'std': np.std(bond_lengths),
                    'min': np.min(bond_lengths),
                    'max': np.max(bond_lengths),
                    'count': len(bond_lengths)
                }
                return stats
        except Exception as e:
            print(f"分析键长失败: {e}")

        return None

    @staticmethod
    def compare_structures(initial_cif, relaxed_cif):
        """比较两个结构"""
        try:
            from ase.io import read

            atoms_initial = read(initial_cif, format='cif')
            atoms_relaxed = read(relaxed_cif, format='cif')

            # 计算RMSD（需要原子顺序一致）
            if len(atoms_initial) == len(atoms_relaxed):
                rmsd = np.sqrt(np.mean((atoms_initial.positions - atoms_relaxed.positions) ** 2))
            else:
                rmsd = None

            # 体积变化
            vol_initial = atoms_initial.get_volume()
            vol_relaxed = atoms_relaxed.get_volume()
            vol_change = (vol_relaxed - vol_initial) / vol_initial * 100

            return {
                'rmsd': rmsd,
                'volume_change_percent': vol_change,
                'initial_volume': vol_initial,
                'relaxed_volume': vol_relaxed
            }
        except Exception as e:
            print(f"比较结构失败: {e}")
            return None


# ==================== 监控系统 ====================
class JobMonitor:
    """作业监控器"""

    def __init__(self):
        self.jobs = {}

    def add_job(self, name, status="等待"):
        """添加作业"""
        self.jobs[name] = {
            'status': status,
            'start_time': None,
            'end_time': None,
            'duration': None,
            'energy': None
        }

    def update_job(self, name, **kwargs):
        """更新作业状态"""
        if name in self.jobs:
            self.jobs[name].update(kwargs)

    def display_status(self):
        """显示作业状态"""
        print("\n" + "=" * 70)
        print("作业监控面板")
        print("=" * 70)

        status_counts = {}
        for job in self.jobs.values():
            status = job.get('status', '未知')
            status_counts[status] = status_counts.get(status, 0) + 1

        print(f"总作业数: {len(self.jobs)}")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")

        print("\n详细状态:")
        print("-" * 70)
        for name, info in self.jobs.items():
            energy_str = f"能量: {info.get('energy', 'N/A'):.6f}" if info.get('energy') else "能量: N/A"
            print(f"{name:20s} {info.get('status', '未知'):10s} {energy_str}")

        print("=" * 70)


# ==================== 主控制系统 ====================
class RelaxationController:
    """弛豫控制器"""

    def __init__(self):
        self.monitor = JobMonitor()
        self.results = []

    def process_structure(self, cif_file):
        """处理单个结构"""
        structure_name = cif_file.stem
        print(f"\n{'=' * 60}")
        print(f"处理结构: {structure_name}")
        print(f"{'=' * 60}")

        # 添加到监控器
        self.monitor.add_job(structure_name, "准备")

        # 创建工作目录
        work_dir = Config.WORK_DIR / structure_name
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. 转换CIF到POSCAR
        self.monitor.update_job(structure_name, status="转换格式")
        poscar_file = work_dir / "POSCAR"

        if not StructureProcessor.cif_to_poscar(cif_file, poscar_file):
            self.monitor.update_job(structure_name, status="转换失败")
            return False

        # 2. 生成DFTB+输入文件
        self.monitor.update_job(structure_name, status="生成输入")
        if not DFTBInputGenerator.generate_input(structure_name, work_dir):
            self.monitor.update_job(structure_name, status="输入生成失败")
            return False

        # 3. 运行DFTB+计算
        self.monitor.update_job(structure_name, status="计算中")
        success, elapsed = DFTBCalculator.run_calculation(structure_name, work_dir)

        if not success:
            self.monitor.update_job(structure_name, status="计算失败")
            return False

        # 4. 提取结果
        self.monitor.update_job(structure_name, status="提取结果")
        results = ResultProcessor.extract_results(work_dir, structure_name)

        if results['success']:
            self.monitor.update_job(
                structure_name,
                status="完成",
                energy=results['energy'],
                duration=elapsed
            )

            # 保存结果
            result_entry = {
                'structure': structure_name,
                'success': True,
                'input_cif': str(cif_file),
                'output_cif': str(results['output_cif']),
                'energy': results['energy'],
                'duration': elapsed,
                'work_dir': str(work_dir)
            }
            self.results.append(result_entry)

            print(f"✓ 结构 {structure_name} 弛豫完成")
            return True
        else:
            self.monitor.update_job(structure_name, status="结果提取失败")
            print(f"✗ 结构 {structure_name} 处理失败")
            return False

    def analyze_results(self):
        """分析所有结果"""
        print(f"\n{'=' * 60}")
        print("结果分析")
        print(f"{'=' * 60}")

        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r.get('success', False)]

        print(f"总处理数: {len(self.results)}")
        print(f"成功数: {len(successful)}")
        print(f"失败数: {len(failed)}")

        if successful:
            # 计算平均能量变化（如果有初始能量）
            energies = [r['energy'] for r in successful if r['energy']]
            if energies:
                avg_energy = np.mean(energies)
                print(f"\n平均总能量: {avg_energy:.6f} Hartree")

            # 计算平均耗时
            durations = [r['duration'] for r in successful if r.get('duration')]
            if durations:
                avg_duration = np.mean(durations)
                print(f"平均计算时间: {avg_duration:.1f} 秒")

        # 分析结构变化
        if successful and len(successful) > 0:
            print(f"\n结构变化分析:")

            # 分析第一个成功结构的键长
            first_result = successful[0]
            if 'output_cif' in first_result:
                cif_file = Path(first_result['output_cif'])
                if cif_file.exists():
                    bond_stats = StructureAnalyzer.analyze_bond_lengths(cif_file)
                    if bond_stats:
                        print(f"弛豫后C-H键长统计:")
                        print(f"  平均: {bond_stats['mean']:.4f} Å")
                        print(f"  标准差: {bond_stats['std']:.4f} Å")
                        print(f"  范围: {bond_stats['min']:.4f} - {bond_stats['max']:.4f} Å")

        return successful, failed

    def generate_report(self):
        """生成详细报告"""
        report_file = Config.LOG_DIR / f"relaxation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_file, 'w') as f:
            f.write("DFTB+结构弛豫报告\n")
            f.write("=" * 70 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理目录: {Config.INPUT_CIF_DIR}\n")
            f.write(f"输出目录: {Config.OUTPUT_CIF_DIR}\n")
            f.write("=" * 70 + "\n\n")

            successful = [r for r in self.results if r['success']]
            failed = [r for r in self.results if not r.get('success', False)]

            f.write(f"统计摘要:\n")
            f.write(f"  总结构数: {len(self.results)}\n")
            f.write(f"  成功数: {len(successful)}\n")
            f.write(f"  失败数: {len(failed)}\n\n")

            f.write("成功结构详情:\n")
            f.write("-" * 70 + "\n")
            for r in successful:
                f.write(f"结构: {r['structure']}\n")
                f.write(f"  输入文件: {r['input_cif']}\n")
                f.write(f"  输出文件: {r['output_cif']}\n")
                if r['energy']:
                    f.write(f"  总能量: {r['energy']:.6f} Hartree\n")
                if r.get('duration'):
                    f.write(f"  计算时间: {r['duration']:.1f} 秒\n")
                f.write("\n")

            if failed:
                f.write("失败结构:\n")
                f.write("-" * 70 + "\n")
                for r in failed:
                    f.write(f"结构: {r.get('structure', 'Unknown')}\n")
                f.write("\n")

            f.write("DFTB+参数设置:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  最大步数: {Config.MAX_STEPS}\n")
            f.write(f"  力收敛标准: {Config.FORCE_TOL}\n")
            f.write(f"  应力收敛标准: {Config.STRESS_TOL}\n")
            f.write(f"  k点网格: {Config.KPOINTS}\n")
            f.write(f"  参数集: {Config.SKF_DIR}\n")

        print(f"详细报告已保存: {report_file}")
        return report_file

    def run_batch(self):
        """批量运行所有结构"""
        print(f"\n开始批量弛豫处理")
        print(f"输入目录: {Config.INPUT_CIF_DIR}")
        print(f"输出目录: {Config.OUTPUT_CIF_DIR}")
        print(f"工作目录: {Config.WORK_DIR}")

        # 获取所有CIF文件
        cif_files = list(Config.INPUT_CIF_DIR.glob("*.cif"))
        if not cif_files:
            print(f"错误: 在 {Config.INPUT_CIF_DIR} 中未找到CIF文件")
            return

        print(f"找到 {len(cif_files)} 个CIF文件")

        # 处理每个文件
        for i, cif_file in enumerate(cif_files, 1):
            print(f"\n[{i}/{len(cif_files)}] ", end="")
            self.process_structure(cif_file)

            # 显示当前状态
            if i % 5 == 0 or i == len(cif_files):
                self.monitor.display_status()

        # 分析结果
        print(f"\n处理完成!")
        successful, failed = self.analyze_results()

        # 生成报告
        report_file = self.generate_report()

        # 显示最终状态
        self.monitor.display_status()

        print(f"\n处理完成!")
        print(f"成功: {len(successful)} 个结构")
        print(f"失败: {len(failed)} 个结构")
        print(f"报告: {report_file}")


# ==================== 命令行界面 ====================
def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("DFTB+ 结构弛豫自动化系统")
    print("版本 1.0")
    print("=" * 70)

    # 检查参数
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "setup":
            print("运行系统设置...")
            SystemSetup.setup_directories()
            SystemSetup.setup_environment()
            return

        elif command == "check":
            print("检查系统状态...")
            SystemSetup.check_dependencies()
            return

        elif command == "monitor":
            print("启动监控模式...")
            controller = RelaxationController()
            while True:
                controller.monitor.display_status()
                print("\n按Ctrl+C退出监控")
                try:
                    time.sleep(10)
                except KeyboardInterrupt:
                    print("\n监控结束")
                    break
            return

    # 正常运行模式
    print("模式: 批量弛豫")

    # 系统检查
    if not SystemSetup.check_dependencies():
        print("系统检查失败，请安装必要依赖")
        return

    # 设置目录和环境
    SystemSetup.setup_directories()
    SystemSetup.setup_environment()

    # 创建并运行控制器
    controller = RelaxationController()

    try:
        controller.run_batch()
    except KeyboardInterrupt:
        print("\n\n用户中断处理")
        print("生成中断报告...")
        controller.generate_report()
        controller.monitor.display_status()
    except Exception as e:
        print(f"\n处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


# ==================== 入口点 ====================
if __name__ == "__main__":
    main()
