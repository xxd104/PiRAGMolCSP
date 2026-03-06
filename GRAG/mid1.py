import os
import sys
import subprocess
import shutil
import time
from pathlib import Path


class DFTBRelaxation:
    def __init__(self):
        # 设置路径
        self.input_dir = Path("/home/nyx/GRAG/low_cifs/CH4")
        self.output_dir = Path("/home/nyx/GRAG/mid_cifs/CH4")
        self.dftb_executable = Path("/home/nyx/GRAG/dftbplus/dftbplus-23.1.x86_64-linux/bin/dftb+")
        self.slako_dir = Path("/home/nyx/GRAG/dftbplus/slakos/3ob-3-1")

        # 检查路径是否存在
        self._check_paths()

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _check_paths(self):
        """检查所有必需的路径和文件是否存在"""
        errors = []

        # 检查输入目录
        if not self.input_dir.exists():
            errors.append(f"输入目录不存在: {self.input_dir}")
        else:
            cif_files = list(self.input_dir.glob("*.cif"))
            if not cif_files:
                errors.append(f"输入目录中没有找到.cif文件: {self.input_dir}")
            else:
                print(f"找到 {len(cif_files)} 个.cif文件")

        # 检查DFTB+可执行文件
        if not self.dftb_executable.exists():
            errors.append(f"DFTB+可执行文件不存在: {self.dftb_executable}")
        elif not os.access(self.dftb_executable, os.X_OK):
            errors.append(f"DFTB+可执行文件没有执行权限: {self.dftb_executable}")

        # 检查Slater-Koster参数目录
        if not self.slako_dir.exists():
            errors.append(f"Slater-Koster参数目录不存在: {self.slako_dir}")
        else:
            # 检查必要的SK文件
            required_sk = ["C-C.skf", "C-H.skf", "H-H.skf"]
            for sk_file in required_sk:
                if not (self.slako_dir / sk_file).exists():
                    errors.append(f"缺少必要的SK文件: {sk_file}")

        # 如果有错误，抛出异常
        if errors:
            error_msg = "初始化检查失败:\n" + "\n".join(errors)
            raise RuntimeError(error_msg)

    def create_simple_input_file(self, gen_filename, temp_dir):
        """创建最简单的DFTB+输入文件"""
        input_file = temp_dir / "dftb_in.hsd"

        # 非常简单的输入文件，避免语法错误
        input_content = f"""Geometry = GenFormat {{
  <<< "{gen_filename}"
}}

Hamiltonian = DFTB {{
  Scc = Yes
  SCCTolerance = 1.0E-5
  MaxSCCIterations = 50

  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{self.slako_dir}/"
    Separator = "-"
    Suffix = ".skf"
  }}

  MaxAngularMomentum {{
    H = "s"
    C = "p"
  }}
}}
"""

        with open(input_file, 'w') as f:
            f.write(input_content)

        print(f"创建输入文件: {input_file}")
        return input_file

    def run_dftb_simple(self, temp_dir):
        """简单运行DFTB+"""
        original_dir = os.getcwd()
        try:
            # 切换到临时目录
            os.chdir(temp_dir)
            print(f"切换到目录: {temp_dir}")

            # 检查文件
            print("目录内容:")
            for f in temp_dir.iterdir():
                print(f"  {f.name}")

            # 运行DFTB+
            cmd = [str(self.dftb_executable)]
            env = os.environ.copy()
            env['DFTB_SLATERKOSTERDIR'] = str(self.slako_dir)

            print(f"运行命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=60
            )

            print(f"返回码: {result.returncode}")

            # 保存输出
            with open("stdout.log", "w") as f:
                f.write(result.stdout)
            with open("stderr.log", "w") as f:
                f.write(result.stderr)

            # 显示重要信息
            if result.returncode != 0:
                print("DFTB+运行失败")
                print("标准错误:")
                print(result.stderr[:500])
                return False, result.stdout, result.stderr
            else:
                print("DFTB+运行成功")
                return True, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            print("DFTB+超时")
            return False, "", "Timeout"
        except Exception as e:
            print(f"运行DFTB+时出错: {str(e)}")
            return False, "", str(e)
        finally:
            os.chdir(original_dir)

    def test_specific_file(self, filename="CH4_candidate_97_relaxed.cif"):
        """测试特定文件"""
        print(f"\n{'=' * 60}")
        print(f"测试文件: {filename}")
        print(f"{'=' * 60}")

        try:
            # 读取CIF文件
            from ase.io import read, write

            cif_path = self.input_dir / filename
            if not cif_path.exists():
                print(f"文件不存在: {cif_path}")
                return

            atoms = read(cif_path)
            print(f"结构信息: {len(atoms)} 个原子")
            print(f"晶胞: {atoms.get_cell()}")

            # 创建临时目录
            temp_dir = Path("test_dftb")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir()

            # 保存为GEN格式
            gen_file = temp_dir / "input.gen"

            # 写入GEN文件
            with open(gen_file, 'w') as f:
                f.write(f"{len(atoms)} S\n")
                elements = list(set(atoms.get_chemical_symbols()))
                f.write(" ".join(elements) + "\n")

                for i, atom in enumerate(atoms):
                    symbol = atom.symbol
                    elem_idx = elements.index(symbol) + 1
                    pos = atom.position
                    f.write(f"{i + 1} {elem_idx} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

                # 晶胞
                cell = atoms.get_cell()
                f.write("0.0 0.0 0.0\n")
                for j in range(3):
                    f.write(f"{cell[j][0]:.6f} {cell[j][1]:.6f} {cell[j][2]:.6f}\n")

            print(f"创建GEN文件: {gen_file}")

            # 创建输入文件
            input_file = self.create_simple_input_file("input.gen", temp_dir)

            # 运行DFTB+
            success, stdout, stderr = self.run_dftb_simple(temp_dir)

            if success:
                print("\n✓ DFTB+运行成功!")
                # 检查输出文件
                output_files = list(temp_dir.glob("*"))
                print(f"生成的输出文件:")
                for f in output_files:
                    print(f"  {f.name} ({f.stat().st_size} 字节)")
            else:
                print("\n✗ DFTB+运行失败")
                print(f"错误信息:")
                print(stderr[:1000])

            # 保存测试结果
            result_file = self.output_dir / f"{filename}_test_result.txt"
            with open(result_file, 'w') as f:
                f.write(f"测试文件: {filename}\n")
                f.write(f"测试时间: {time.ctime()}\n")
                f.write(f"成功: {success}\n")
                f.write(f"\n标准输出:\n{stdout[:5000]}\n")
                f.write(f"\n标准错误:\n{stderr[:5000]}\n")

            print(f"\n测试结果保存到: {result_file}")

            return success

        except Exception as e:
            print(f"\n测试过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def check_dftb_installation(self):
        """检查DFTB+安装"""
        print(f"\n{'=' * 60}")
        print("检查DFTB+安装")
        print(f"{'=' * 60}")

        # 检查可执行文件
        print(f"DFTB+可执行文件: {self.dftb_executable}")
        if not self.dftb_executable.exists():
            print("✗ 可执行文件不存在")
            return False

        # 检查版本
        try:
            result = subprocess.run(
                [str(self.dftb_executable), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            print(f"DFTB+版本信息:")
            print(result.stdout[:200])
        except:
            print("✗ 无法获取DFTB+版本")
            return False

        # 检查参数文件
        print(f"\n参数文件目录: {self.slako_dir}")
        required_files = ["C-C.skf", "C-H.skf", "H-H.skf"]
        for f in required_files:
            file_path = self.slako_dir / f
            if file_path.exists():
                print(f"✓ {f}")
            else:
                print(f"✗ {f} 不存在")
                return False

        # 简单测试运行
        print(f"\n运行简单测试...")
        test_dir = Path("dftb_test")
        test_dir.mkdir(exist_ok=True)

        # 创建最简单的测试输入
        test_input = test_dir / "test.hsd"
        with open(test_input, 'w') as f:
            f.write("""Geometry = {
  Type = XYZ
  <<< "
2
Test
H 0.0 0.0 0.0
H 1.0 0.0 0.0
"
}

Hamiltonian = DFTB {
  Scc = No
  SlaterKosterFiles = Type2FileNames {
    Prefix = "%s/"
    Separator = "-"
    Suffix = ".skf"
  }
  MaxAngularMomentum {
    H = "s"
  }
}
""" % str(self.slako_dir))

        try:
            os.chdir(test_dir)
            result = subprocess.run(
                [str(self.dftb_executable)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print("✓ DFTB+安装正常")
                os.chdir("..")
                shutil.rmtree(test_dir)
                return True
            else:
                print("✗ DFTB+测试运行失败")
                print(f"错误信息:\n{result.stderr[:500]}")
                os.chdir("..")
                return False

        except Exception as e:
            print(f"✗ 测试运行出错: {str(e)}")
            if 'test_dir' in locals():
                os.chdir("..")
            return False

    def process_all_files_basic(self):
        """基本处理所有文件"""
        print(f"\n{'=' * 60}")
        print("基本处理所有CIF文件")
        print(f"{'=' * 60}")

        # 检查安装
        if not self.check_dftb_installation():
            print("✗ DFTB+安装检查失败，停止处理")
            return

        # 获取所有CIF文件
        cif_files = list(self.input_dir.glob("*.cif"))
        if not cif_files:
            print("没有找到CIF文件")
            return

        print(f"找到 {len(cif_files)} 个CIF文件")

        success_count = 0

        for i, cif_file in enumerate(cif_files, 1):
            print(f"\n[{i}/{len(cif_files)}] 处理文件: {cif_file.name}")

            try:
                # 创建临时目录
                temp_dir = Path(f"temp_{cif_file.stem}")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                temp_dir.mkdir()

                # 转换CIF为GEN
                from ase.io import read, write
                atoms = read(cif_file)

                gen_file = temp_dir / "structure.gen"
                with open(gen_file, 'w') as f:
                    f.write(f"{len(atoms)} S\n")
                    elements = list(set(atoms.get_chemical_symbols()))
                    f.write(" ".join(elements) + "\n")

                    for j, atom in enumerate(atoms):
                        symbol = atom.symbol
                        elem_idx = elements.index(symbol) + 1
                        pos = atom.position
                        f.write(f"{j + 1} {elem_idx} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

                    cell = atoms.get_cell()
                    f.write("0.0 0.0 0.0\n")
                    for j in range(3):
                        f.write(f"{cell[j][0]:.6f} {cell[j][1]:.6f} {cell[j][2]:.6f}\n")

                # 创建输入文件
                input_file = temp_dir / "dftb_in.hsd"
                with open(input_file, 'w') as f:
                    f.write(f"""Geometry = GenFormat {{
  <<< "structure.gen"
}}

Driver = GeometryOptimization {{
  MaxSteps = 100
  MaxForceComponent = 0.01
}}

Hamiltonian = DFTB {{
  Scc = Yes
  SCCTolerance = 1.0E-5
  MaxSCCIterations = 100

  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{self.slako_dir}/"
    Separator = "-"
    Suffix = ".skf"
  }}

  MaxAngularMomentum {{
    H = "s"
    C = "p"
  }}
}}

Options = {{
  WriteDetailedOut = Yes
  WriteChargesAsText = Yes
}}
""")

                # 运行DFTB+
                original_dir = os.getcwd()
                os.chdir(temp_dir)

                try:
                    result = subprocess.run(
                        [str(self.dftb_executable)],
                        capture_output=True,
                        text=True,
                        timeout=300,
                        env={**os.environ, 'DFTB_SLATERKOSTERDIR': str(self.slako_dir)}
                    )

                    if result.returncode == 0:
                        # 保存结果
                        output_cif = self.output_dir / f"{cif_file.stem}_processed.cif"
                        write(str(output_cif), atoms, format='cif')
                        print(f"✓ 成功处理: {cif_file.name} -> {output_cif.name}")
                        success_count += 1
                    else:
                        # 即使失败也保存原始结构
                        output_cif = self.output_dir / f"{cif_file.stem}_original.cif"
                        write(str(output_cif), atoms, format='cif')
                        print(f"✗ DFTB+失败，保存原始结构: {cif_file.name}")

                        # 保存错误日志
                        error_log = self.output_dir / f"{cif_file.stem}_error.log"
                        with open(error_log, 'w') as f:
                            f.write(f"文件: {cif_file.name}\n")
                            f.write(f"DFTB+返回码: {result.returncode}\n")
                            f.write(f"标准错误:\n{result.stderr[:2000]}\n")

                except subprocess.TimeoutExpired:
                    print(f"✗ 超时: {cif_file.name}")
                    # 保存原始结构
                    output_cif = self.output_dir / f"{cif_file.stem}_timeout.cif"
                    write(str(output_cif), atoms, format='cif')

                finally:
                    os.chdir(original_dir)

                # 清理临时目录
                shutil.rmtree(temp_dir)

            except Exception as e:
                print(f"✗ 处理过程中出错: {cif_file.name} - {str(e)}")

        # 输出总结
        print(f"\n{'=' * 60}")
        print("处理完成!")
        print(f"{'=' * 60}")
        print(f"成功处理: {success_count}/{len(cif_files)}")

        summary_file = self.output_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"处理时间: {time.ctime()}\n")
            f.write(f"总文件数: {len(cif_files)}\n")
            f.write(f"成功处理: {success_count}\n")

        print(f"总结保存到: {summary_file}")


def main():
    print("DFTB+结构弛豫工具")
    print("=" * 60)

    # 创建工作流实例
    workflow = DFTBRelaxation()

    # 选择操作
    print("\n请选择操作:")
    print("1. 检查DFTB+安装")
    print("2. 测试特定文件 (CH4_candidate_97_relaxed.cif)")
    print("3. 基本处理所有文件")

    try:
        choice = int(input("请输入选择 (1-3): "))
    except:
        print("无效输入，使用默认选项1")
        choice = 1

    if choice == 1:
        workflow.check_dftb_installation()
    elif choice == 2:
        workflow.test_specific_file()
    elif choice == 3:
        workflow.process_all_files_basic()
    else:
        print("无效选择，退出")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {str(e)}")
        import traceback

        traceback.print_exc()