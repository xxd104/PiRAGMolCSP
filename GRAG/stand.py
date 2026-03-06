import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


class CIFSymmetryExpander:
    """
    将包含独立原子的CIF文件扩展为具有完整对称性的晶体结构
    """

    def __init__(self, tolerance: float = 0.01):
        """
        初始化CIF对称性扩展器

        Args:
            tolerance: 原子位置去重时的容差
        """
        self.tolerance = tolerance

        # 常见空间群的对称操作生成器
        self.symmetry_generators = {
            'P1': self._generate_P1_symops,
            'P-1': self._generate_P_1_symops,
            'P2': self._generate_P2_symops,
            'P21': self._generate_P21_symops,
            'C2': self._generate_C2_symops,
            'P222': self._generate_P222_symops,
            'P21212': self._generate_P21212_symops,
            'P212121': self._generate_P212121_symops,
            'C222': self._generate_C222_symops,
            'F222': self._generate_F222_symops,
            'I222': self._generate_I222_symops,
            'I212121': self._generate_I212121_symops,
            'P4': self._generate_P4_symops,
            'P41': self._generate_P41_symops,
            'P42': self._generate_P42_symops,
            'P43': self._generate_P43_symops,
            'I4': self._generate_I4_symops,
            'I-4': self._generate_I_4_symops,  # 你的例子中的空间群
            'P4/m': self._generate_P4_m_symops,
            'P42/m': self._generate_P42_m_symops,
            'P4/n': self._generate_P4_n_symops,
            'P42/n': self._generate_P42_n_symops,
            'I4/m': self._generate_I4_m_symops,
            'I41/a': self._generate_I41_a_symops,
            'R3': self._generate_R3_symops,
            'R-3': self._generate_R_3_symops,
            'P312': self._generate_P312_symops,
            'P321': self._generate_P321_symops,
            'P3_121': self._generate_P3_121_symops,
            'P3_221': self._generate_P3_221_symops,
            'P6': self._generate_P6_symops,
            'P61': self._generate_P61_symops,
            'P62': self._generate_P62_symops,
            'P63': self._generate_P63_symops,
            'P64': self._generate_P64_symops,
            'P65': self._generate_P65_symops,
            'P6/m': self._generate_P6_m_symops,
            'P63/m': self._generate_P63_m_symops,
            'P23': self._generate_P23_symops,
            'F23': self._generate_F23_symops,
            'I23': self._generate_I23_symops,
            'P213': self._generate_P213_symops,
            'I213': self._generate_I213_symops,
            'Pm-3m': self._generate_Pm_3m_symops,
            'Fm-3m': self._generate_Fm_3m_symops,
            'Im-3m': self._generate_Im_3m_symops,
            'Fd-3m': self._generate_Fd_3m_symops,
        }

    def _generate_I_4_symops(self) -> List[str]:
        """生成I-4空间群的对称操作"""
        return [
            'x, y, z',
            '-x, -y, z',
            '-y, x, -z',
            'y, -x, -z',
            'x+1/2, y+1/2, z+1/2',
            '-x+1/2, -y+1/2, z+1/2',
            '-y+1/2, x+1/2, -z+1/2',
            'y+1/2, -x+1/2, -z+1/2',
        ]

    def _generate_P1_symops(self) -> List[str]:
        """生成P1空间群的对称操作"""
        return ['x, y, z']

    def _generate_P_1_symops(self) -> List[str]:
        """生成P-1空间群的对称操作"""
        return ['x, y, z', '-x, -y, -z']

    def _generate_P2_symops(self) -> List[str]:
        """生成P2空间群的对称操作"""
        return ['x, y, z', '-x, y, -z']

    def _generate_P21_symops(self) -> List[str]:
        """生成P21空间群的对称操作"""
        return ['x, y, z', '-x, y+1/2, -z']

    def _generate_C2_symops(self) -> List[str]:
        """生成C2空间群的对称操作"""
        return [
            'x, y, z',
            '-x, y, -z',
            'x+1/2, y+1/2, z',
            '-x+1/2, y+1/2, -z'
        ]

    def _generate_P222_symops(self) -> List[str]:
        """生成P222空间群的对称操作"""
        return [
            'x, y, z',
            '-x, -y, z',
            '-x, y, -z',
            'x, -y, -z'
        ]

    def _generate_P21212_symops(self) -> List[str]:
        """生成P21212空间群的对称操作"""
        return [
            'x, y, z',
            '-x, -y, z',
            '-x+1/2, y+1/2, -z',
            'x+1/2, -y+1/2, -z'
        ]

    def _generate_P212121_symops(self) -> List[str]:
        """生成P212121空间群的对称操作"""
        return [
            'x, y, z',
            '-x+1/2, -y, z+1/2',
            '-x, y+1/2, -z+1/2',
            'x+1/2, -y+1/2, -z'
        ]

    def _generate_C222_symops(self) -> List[str]:
        """生成C222空间群的对称操作"""
        return [
            'x, y, z',
            '-x, -y, z',
            '-x, y, -z',
            'x, -y, -z',
            'x+1/2, y+1/2, z',
            '-x+1/2, -y+1/2, z',
            '-x+1/2, y+1/2, -z',
            'x+1/2, -y+1/2, -z'
        ]

    def _generate_F222_symops(self) -> List[str]:
        """生成F222空间群的对称操作"""
        symops = []
        for tx in [0, 0.5]:
            for ty in [0, 0.5]:
                for tz in [0, 0.5]:
                    if (tx + ty + tz) % 1 == 0:  # 只取F心条件
                        symops.extend([
                            f'x+{tx}, y+{ty}, z+{tz}',
                            f'-x+{tx}, -y+{ty}, z+{tz}',
                            f'-x+{tx}, y+{ty}, -z+{tz}',
                            f'x+{tx}, -y+{ty}, -z+{tz}'
                        ])
        return symops

    def _generate_I222_symops(self) -> List[str]:
        """生成I222空间群的对称操作"""
        symops = []
        for t in [0, 0.5]:
            symops.extend([
                f'x+{t}, y+{t}, z+{t}',
                f'-x+{t}, -y+{t}, z+{t}',
                f'-x+{t}, y+{t}, -z+{t}',
                f'x+{t}, -y+{t}, -z+{t}'
            ])
        return symops

    def _generate_I212121_symops(self) -> List[str]:
        """生成I212121空间群的对称操作"""
        return [
            'x, y, z',
            '-x+1/2, -y, z+1/2',
            '-x, y+1/2, -z+1/2',
            'x+1/2, -y+1/2, -z',
            'x+1/2, y+1/2, z+1/2',
            '-x, -y+1/2, z',
            '-x+1/2, y, -z',
            'x, -y, -z+1/2'
        ]

    def _generate_P4_symops(self) -> List[str]:
        """生成P4空间群的对称操作"""
        return [
            'x, y, z',
            '-x, -y, z',
            '-y, x, z',
            'y, -x, z'
        ]

    def _generate_P41_symops(self) -> List[str]:
        """生成P41空间群的对称操作"""
        return [
            'x, y, z',
            '-x, -y, z+1/2',
            '-y, x, z+1/4',
            'y, -x, z+3/4'
        ]

    def _generate_P42_symops(self) -> List[str]:
        """生成P42空间群的对称操作"""
        return [
            'x, y, z',
            '-x, -y, z',
            '-y, x, z+1/2',
            'y, -x, z+1/2'
        ]

    def _generate_P43_symops(self) -> List[str]:
        """生成P43空间群的对称操作"""
        return [
            'x, y, z',
            '-x, -y, z+1/2',
            '-y, x, z+3/4',
            'y, -x, z+1/4'
        ]

    def _generate_I4_symops(self) -> List[str]:
        """生成I4空间群的对称操作"""
        symops = []
        for t in [0, 0.5]:
            symops.extend([
                f'x+{t}, y+{t}, z+{t}',
                f'-x+{t}, -y+{t}, z+{t}',
                f'-y+{t}, x+{t}, z+{t}',
                f'y+{t}, -x+{t}, z+{t}'
            ])
        return symops

    def _generate_P4_m_symops(self) -> List[str]:
        """生成P4/m空间群的对称操作"""
        symops = self._generate_P4_symops()
        symops += ['-x, -y, -z', 'x, y, -z', 'y, -x, -z', '-y, x, -z']
        return symops

    def _generate_P42_m_symops(self) -> List[str]:
        """生成P42/m空间群的对称操作"""
        return [
            'x, y, z',
            '-x, -y, z',
            '-y, x, z+1/2',
            'y, -x, z+1/2',
            '-x, -y, -z',
            'x, y, -z',
            'y, -x, -z+1/2',
            '-y, x, -z+1/2'
        ]

    def _generate_P4_n_symops(self) -> List[str]:
        """生成P4/n空间群的对称操作"""
        return [
            'x, y, z',
            '-x, -y, z',
            '-y+1/2, x+1/2, z',
            'y+1/2, -x+1/2, z',
            '-x+1/2, -y+1/2, -z',
            'x+1/2, y+1/2, -z',
            'y, -x, -z',
            '-y, x, -z'
        ]

    def _generate_P42_n_symops(self) -> List[str]:
        """生成P42/n空间群的对称操作"""
        return [
            'x, y, z',
            '-x, -y, z',
            '-y+1/2, x+1/2, z+1/2',
            'y+1/2, -x+1/2, z+1/2',
            '-x+1/2, -y+1/2, -z',
            'x+1/2, y+1/2, -z',
            'y, -x, -z+1/2',
            '-y, x, -z+1/2'
        ]

    def _generate_I4_m_symops(self) -> List[str]:
        """生成I4/m空间群的对称操作"""
        symops = []
        for t in [0, 0.5]:
            symops.extend([
                f'x+{t}, y+{t}, z+{t}',
                f'-x+{t}, -y+{t}, z+{t}',
                f'-y+{t}, x+{t}, z+{t}',
                f'y+{t}, -x+{t}, z+{t}',
                f'-x+{t}, -y+{t}, -z+{t}',
                f'x+{t}, y+{t}, -z+{t}',
                f'y+{t}, -x+{t}, -z+{t}',
                f'-y+{t}, x+{t}, -z+{t}'
            ])
        return symops

    def _generate_I41_a_symops(self) -> List[str]:
        """生成I41/a空间群的对称操作"""
        return [
            'x, y, z',
            '-x+1/2, -y, z+1/2',
            '-y+1/4, x+1/4, z+1/4',
            'y+1/4, -x+3/4, z+3/4',
            '-x, -y, -z',
            'x+1/2, y, -z+1/2',
            'y+3/4, -x+3/4, -z+3/4',
            '-y+3/4, x+1/4, -z+1/4',
            'x+1/2, y+1/2, z+1/2',
            '-x, -y+1/2, z',
            '-y+3/4, x+3/4, z+3/4',
            'y+3/4, -x+1/4, z+1/4',
            '-x+1/2, -y+1/2, -z+1/2',
            'x, y+1/2, -z',
            'y+1/4, -x+1/4, -z+1/4',
            '-y+1/4, x+3/4, -z+3/4'
        ]

    def _generate_R3_symops(self) -> List[str]:
        """生成R3空间群的对称操作"""
        return [
            'x, y, z',
            'z, x, y',
            'y, z, x'
        ]

    def _generate_R_3_symops(self) -> List[str]:
        """生成R-3空间群的对称操作"""
        symops = self._generate_R3_symops()
        symops += ['-x, -y, -z', '-z, -x, -y', '-y, -z, -x']
        return symops

    def _generate_P312_symops(self) -> List[str]:
        """生成P312空间群的对称操作"""
        return [
            'x, y, z',
            'y, x, -z',
            '-y, x-y, z',
            '-x, y-x, -z',
            'x-y, -x, z',
            'y-x, -y, -z'
        ]

    def _generate_P321_symops(self) -> List[str]:
        """生成P321空间群的对称操作"""
        return [
            'x, y, z',
            '-y, x-y, z',
            '-x+y, -x, z',
            '-y, -x, -z',
            'x-y, y, -z',
            'x, x-y, -z'
        ]

    def _generate_P3_121_symops(self) -> List[str]:
        """生成P3_121空间群的对称操作"""
        return [
            'x, y, z',
            '-y, x-y, z+2/3',
            '-x+y, -x, z+1/3',
            '-y, -x, -z+2/3',
            'x-y, y, -z+1/3',
            'x, x-y, -z'
        ]

    def _generate_P3_221_symops(self) -> List[str]:
        """生成P3_221空间群的对称操作"""
        return [
            'x, y, z',
            '-y, x-y, z+1/3',
            '-x+y, -x, z+2/3',
            '-y, -x, -z+1/3',
            'x-y, y, -z+2/3',
            'x, x-y, -z'
        ]

    def _generate_P6_symops(self) -> List[str]:
        """生成P6空间群的对称操作"""
        return [
            'x, y, z',
            '-y, x-y, z',
            '-x+y, -x, z',
            '-x, -y, z',
            'y, -x+y, z',
            'x-y, x, z'
        ]

    def _generate_P61_symops(self) -> List[str]:
        """生成P61空间群的对称操作"""
        return [
            'x, y, z',
            '-y, x-y, z+1/6',
            '-x+y, -x, z+1/3',
            '-x, -y, z+1/2',
            'y, -x+y, z+2/3',
            'x-y, x, z+5/6'
        ]

    def _generate_P62_symops(self) -> List[str]:
        """生成P62空间群的对称操作"""
        return [
            'x, y, z',
            '-y, x-y, z+1/3',
            '-x+y, -x, z+2/3',
            '-x, -y, z',
            'y, -x+y, z+1/3',
            'x-y, x, z+2/3'
        ]

    def _generate_P63_symops(self) -> List[str]:
        """生成P63空间群的对称操作"""
        return [
            'x, y, z',
            '-y, x-y, z+1/2',
            '-x+y, -x, z',
            '-x, -y, z+1/2',
            'y, -x+y, z',
            'x-y, x, z+1/2'
        ]

    def _generate_P64_symops(self) -> List[str]:
        """生成P64空间群的对称操作"""
        return [
            'x, y, z',
            '-y, x-y, z+2/3',
            '-x+y, -x, z+1/3',
            '-x, -y, z',
            'y, -x+y, z+2/3',
            'x-y, x, z+1/3'
        ]

    def _generate_P65_symops(self) -> List[str]:
        """生成P65空间群的对称操作"""
        return [
            'x, y, z',
            '-y, x-y, z+5/6',
            '-x+y, -x, z+2/3',
            '-x, -y, z+1/2',
            'y, -x+y, z+1/3',
            'x-y, x, z+1/6'
        ]

    def _generate_P6_m_symops(self) -> List[str]:
        """生成P6/m空间群的对称操作"""
        symops = self._generate_P6_symops()
        symops += [
            '-x, -y, -z',
            'y, -x+y, -z',
            'x-y, x, -z',
            'x, y, -z',
            '-y, x-y, -z',
            '-x+y, -x, -z'
        ]
        return symops

    def _generate_P63_m_symops(self) -> List[str]:
        """生成P63/m空间群的对称操作"""
        return [
            'x, y, z',
            '-y, x-y, z+1/2',
            '-x+y, -x, z',
            '-x, -y, z+1/2',
            'y, -x+y, z',
            'x-y, x, z+1/2',
            '-x, -y, -z',
            'y, -x+y, -z+1/2',
            'x-y, x, -z',
            'x, y, -z+1/2',
            '-y, x-y, -z',
            '-x+y, -x, -z+1/2'
        ]

    def _generate_P23_symops(self) -> List[str]:
        """生成P23空间群的对称操作"""
        return [
            'x, y, z',
            '-x, -y, z',
            '-x, y, -z',
            'x, -y, -z',
            'z, x, y',
            'z, -x, -y',
            '-z, -x, y',
            '-z, x, -y',
            'y, z, x',
            '-y, z, -x',
            'y, -z, -x',
            '-y, -z, x'
        ]

    def _generate_F23_symops(self) -> List[str]:
        """生成F23空间群的对称操作"""
        symops = []
        for tx in [0, 0.5]:
            for ty in [0, 0.5]:
                for tz in [0, 0.5]:
                    if (tx + ty + tz) % 1 == 0:  # F心条件
                        symops.extend([
                            f'x+{tx}, y+{ty}, z+{tz}',
                            f'-x+{tx}, -y+{ty}, z+{tz}',
                            f'-x+{tx}, y+{ty}, -z+{tz}',
                            f'x+{tx}, -y+{ty}, -z+{tz}',
                            f'z+{tx}, x+{ty}, y+{tz}',
                            f'z+{tx}, -x+{ty}, -y+{tz}',
                            f'-z+{tx}, -x+{ty}, y+{tz}',
                            f'-z+{tx}, x+{ty}, -y+{tz}',
                            f'y+{tx}, z+{ty}, x+{tz}',
                            f'-y+{tx}, z+{ty}, -x+{tz}',
                            f'y+{tx}, -z+{ty}, -x+{tz}',
                            f'-y+{tx}, -z+{ty}, x+{tz}'
                        ])
        return symops

    def _generate_I23_symops(self) -> List[str]:
        """生成I23空间群的对称操作"""
        symops = []
        for t in [0, 0.5]:
            symops.extend([
                f'x+{t}, y+{t}, z+{t}',
                f'-x+{t}, -y+{t}, z+{t}',
                f'-x+{t}, y+{t}, -z+{t}',
                f'x+{t}, -y+{t}, -z+{t}',
                f'z+{t}, x+{t}, y+{t}',
                f'z+{t}, -x+{t}, -y+{t}',
                f'-z+{t}, -x+{t}, y+{t}',
                f'-z+{t}, x+{t}, -y+{t}',
                f'y+{t}, z+{t}, x+{t}',
                f'-y+{t}, z+{t}, -x+{t}',
                f'y+{t}, -z+{t}, -x+{t}',
                f'-y+{t}, -z+{t}, x+{t}'
            ])
        return symops

    def _generate_P213_symops(self) -> List[str]:
        """生成P213空间群的对称操作"""
        return [
            'x, y, z',
            '-x+1/2, -y, z+1/2',
            '-x, y+1/2, -z+1/2',
            'x+1/2, -y+1/2, -z',
            'z, x, y',
            'z+1/2, -x+1/2, -y',
            '-z+1/2, -x, y+1/2',
            '-z, x+1/2, -y+1/2',
            'y, z, x',
            '-y, z+1/2, -x+1/2',
            'y+1/2, -z+1/2, -x',
            '-y+1/2, -z, x+1/2'
        ]

    def _generate_I213_symops(self) -> List[str]:
        """生成I213空间群的对称操作"""
        symops = []
        for t in [0, 0.5]:
            symops.extend([
                f'x+{t}, y+{t}, z+{t}',
                f'-x+{t}+1/2, -y+{t}, z+{t}+1/2',
                f'-x+{t}, y+{t}+1/2, -z+{t}+1/2',
                f'x+{t}+1/2, -y+{t}+1/2, -z+{t}',
                f'z+{t}, x+{t}, y+{t}',
                f'z+{t}+1/2, -x+{t}+1/2, -y+{t}',
                f'-z+{t}+1/2, -x+{t}, y+{t}+1/2',
                f'-z+{t}, x+{t}+1/2, -y+{t}+1/2',
                f'y+{t}, z+{t}, x+{t}',
                f'-y+{t}, z+{t}+1/2, -x+{t}+1/2',
                f'y+{t}+1/2, -z+{t}+1/2, -x+{t}',
                f'-y+{t}+1/2, -z+{t}, x+{t}+1/2'
            ])
        return symops

    def _generate_Pm_3m_symops(self) -> List[str]:
        """生成Pm-3m空间群的对称操作"""
        symops = self._generate_P23_symops()
        # 添加反演中心
        additional_symops = []
        for symop in symops:
            # 将每个对称操作取反
            if 'x' in symop and 'y' in symop and 'z' in symop:
                inv_symop = symop.replace('x', '-x').replace('y', '-y').replace('z', '-z')
                # 处理双负号
                inv_symop = inv_symop.replace('--', '')
                additional_symops.append(inv_symop)
        symops += additional_symops
        return list(set(symops))  # 去重

    def _generate_Fm_3m_symops(self) -> List[str]:
        """生成Fm-3m空间群的对称操作"""
        symops = self._generate_F23_symops()
        # 添加反演中心
        additional_symops = []
        for symop in symops:
            if 'x' in symop and 'y' in symop and 'z' in symop:
                inv_symop = symop.replace('x', '-x').replace('y', '-y').replace('z', '-z')
                inv_symop = inv_symop.replace('--', '')
                additional_symops.append(inv_symop)
        symops += additional_symops
        return list(set(symops))

    def _generate_Im_3m_symops(self) -> List[str]:
        """生成Im-3m空间群的对称操作"""
        symops = self._generate_I23_symops()
        # 添加反演中心
        additional_symops = []
        for symop in symops:
            if 'x' in symop and 'y' in symop and 'z' in symop:
                inv_symop = symop.replace('x', '-x').replace('y', '-y').replace('z', '-z')
                inv_symop = inv_symop.replace('--', '')
                additional_symops.append(inv_symop)
        symops += additional_symops
        return list(set(symops))

    def _generate_Fd_3m_symops(self) -> List[str]:
        """生成Fd-3m空间群的对称操作"""
        symops = []
        # 金刚石结构空间群，复杂对称操作
        # 这里提供一个简化版本
        for tx in [0, 0.25, 0.5, 0.75]:
            for ty in [0, 0.25, 0.5, 0.75]:
                for tz in [0, 0.25, 0.5, 0.75]:
                    if (tx + ty + tz) % 0.5 == 0:  # F心条件
                        symops.extend([
                            f'x+{tx}, y+{ty}, z+{tz}',
                            f'-x+{tx}, -y+{ty}, z+{tz}',
                            f'-x+{tx}, y+{ty}, -z+{tz}',
                            f'x+{tx}, -y+{ty}, -z+{tz}',
                            f'z+{tx}, x+{ty}, y+{tz}',
                            f'z+{tx}, -x+{ty}, -y+{tz}',
                            f'-z+{tx}, -x+{ty}, y+{tz}',
                            f'-z+{tx}, x+{ty}, -y+{tz}',
                            f'y+{tx}, z+{ty}, x+{tz}',
                            f'-y+{tx}, z+{ty}, -x+{tz}',
                            f'y+{tx}, -z+{ty}, -x+{tz}',
                            f'-y+{tx}, -z+{ty}, x+{tz}'
                        ])
        return list(set(symops))  # 去重

    def get_symmetry_operations(self, space_group: str) -> List[str]:
        """
        根据空间群符号获取对称操作

        Args:
            space_group: 空间群符号 (如 'P1', 'I-4', 'Fd-3m' 等)

        Returns:
            对称操作字符串列表
        """
        # 清理空间群符号
        space_group = space_group.strip().replace("'", "")

        # 检查是否有生成器
        if space_group in self.symmetry_generators:
            return self.symmetry_generators[space_group]()
        else:
            # 如果没有对应的生成器，尝试使用通用方法或返回恒等操作
            warnings.warn(f"未找到空间群 {space_group} 的对称操作生成器，使用恒等操作")
            return ['x, y, z']

    def apply_symmetry_operation(self, x: float, y: float, z: float,
                                 symop: str) -> Tuple[float, float, float]:
        """
        应用对称操作到原子位置

        Args:
            x, y, z: 原子分数坐标
            symop: 对称操作字符串 (如 'x, y, z', '-x, -y, z+1/2')

        Returns:
            新的分数坐标
        """
        # 解析对称操作
        # 移除空格并分割
        parts = symop.split(',')
        if len(parts) != 3:
            raise ValueError(f"无效的对称操作格式: {symop}")

        # 解析每个坐标变换
        x_expr = parts[0].strip()
        y_expr = parts[1].strip()
        z_expr = parts[2].strip()

        # 应用变换
        new_x = self._evaluate_expression(x_expr, x, y, z)
        new_y = self._evaluate_expression(y_expr, x, y, z)
        new_z = self._evaluate_expression(z_expr, x, y, z)

        # 将坐标规范到 [0, 1) 区间
        new_x = new_x % 1
        new_y = new_y % 1
        new_z = new_z % 1

        return new_x, new_y, new_z

    def _evaluate_expression(self, expr: str, x: float, y: float, z: float) -> float:
        """
        评估表达式，支持简单的坐标变换

        Args:
            expr: 表达式字符串 (如 'x', '-x+1/2', 'y+1/4')
            x, y, z: 原始坐标

        Returns:
            计算后的值
        """
        # 简单的表达式解析
        result = 0.0

        # 检查是否有x项
        if 'x' in expr:
            coeff = 1.0
            if expr.startswith('-x'):
                coeff = -1.0
                expr = expr[2:]
            elif expr.startswith('x'):
                coeff = 1.0
                expr = expr[1:]
            elif expr.startswith('+x'):
                coeff = 1.0
                expr = expr[2:]
            result += coeff * x

        # 检查是否有y项
        if 'y' in expr:
            coeff = 1.0
            if '-y' in expr:
                coeff = -1.0
            elif '+y' in expr:
                coeff = 1.0
            result += coeff * y

        # 检查是否有z项
        if 'z' in expr:
            coeff = 1.0
            if '-z' in expr:
                coeff = -1.0
            elif '+z' in expr:
                coeff = 1.0
            result += coeff * z

        # 处理常数项 (如 +1/2, -1/4)
        import re
        const_matches = re.findall(r'[+-]?\s*\d+/\d+', expr.replace(' ', ''))
        for const in const_matches:
            if '/' in const:
                num, den = const.split('/')
                if num.startswith('+'):
                    num = num[1:]
                value = float(num) / float(den)
                if const.startswith('-'):
                    value = -value
                result += value

        # 处理整数常数
        int_matches = re.findall(r'[+-]?\s*\d+(?![/\d])', expr.replace(' ', ''))
        for const in int_matches:
            value = float(const)
            result += value

        return result

    def are_positions_equivalent(self, pos1: Tuple[float, float, float],
                                 pos2: Tuple[float, float, float]) -> bool:
        """
        检查两个位置是否等效（考虑周期性边界条件）

        Args:
            pos1: 第一个位置 (x, y, z)
            pos2: 第二个位置 (x, y, z)

        Returns:
            如果位置等效则返回True
        """
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2

        # 考虑周期性边界条件
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    x2_shift = x2 + dx
                    y2_shift = y2 + dy
                    z2_shift = z2 + dz

                    dx_val = abs(x1 - x2_shift)
                    dy_val = abs(y1 - y2_shift)
                    dz_val = abs(z1 - z2_shift)

                    if (dx_val < self.tolerance and
                            dy_val < self.tolerance and
                            dz_val < self.tolerance):
                        return True

        return False

    def expand_atoms(self, atoms: List[Dict], space_group: str) -> List[Dict]:
        """
        根据对称操作扩展原子

        Args:
            atoms: 原子列表，每个原子是包含 'label', 'symbol', 'x', 'y', 'z', 'occupancy' 的字典
            space_group: 空间群符号

        Returns:
            扩展后的原子列表
        """
        # 获取对称操作
        symops = self.get_symmetry_operations(space_group)

        # 扩展原子
        expanded_atoms = []
        atom_counters = {}

        for atom in atoms:
            base_symbol = atom['symbol']
            if base_symbol not in atom_counters:
                atom_counters[base_symbol] = 1

            # 对每个对称操作生成新原子
            for symop in symops:
                new_x, new_y, new_z = self.apply_symmetry_operation(
                    atom['x'], atom['y'], atom['z'], symop
                )

                # 创建新原子
                new_atom = {
                    'symbol': base_symbol,
                    'x': new_x,
                    'y': new_y,
                    'z': new_z,
                    'occupancy': atom['occupancy']
                }

                # 检查是否已存在等效原子
                is_duplicate = False
                for existing_atom in expanded_atoms:
                    if (existing_atom['symbol'] == new_atom['symbol'] and
                            self.are_positions_equivalent(
                                (existing_atom['x'], existing_atom['y'], existing_atom['z']),
                                (new_atom['x'], new_atom['y'], new_atom['z'])
                            )):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    # 生成标签
                    label = f"{base_symbol}{atom_counters[base_symbol]}"
                    atom_counters[base_symbol] += 1
                    new_atom['label'] = label
                    expanded_atoms.append(new_atom)

        return expanded_atoms


def parse_cif_file(filepath: str) -> Tuple[Dict, List[Dict], List[str]]:
    """
    解析CIF文件

    Args:
        filepath: CIF文件路径

    Returns:
        cell_params: 晶胞参数字典
        atoms: 原子列表
        original_lines: 原始文件行列表
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    cell_params = {}
    atoms = []
    in_atom_block = False
    atom_labels = []

    for line in lines:
        line = line.strip()

        # 解析晶胞参数
        if line.startswith('_cell_length_a'):
            cell_params['a'] = float(line.split()[1])
        elif line.startswith('_cell_length_b'):
            cell_params['b'] = float(line.split()[1])
        elif line.startswith('_cell_length_c'):
            cell_params['c'] = float(line.split()[1])
        elif line.startswith('_cell_angle_alpha'):
            cell_params['alpha'] = float(line.split()[1])
        elif line.startswith('_cell_angle_beta'):
            cell_params['beta'] = float(line.split()[1])
        elif line.startswith('_cell_angle_gamma'):
            cell_params['gamma'] = float(line.split()[1])

        # 解析空间群
        elif line.startswith('_symmetry_space_group_name_H-M'):
            # 提取空间群符号，处理可能的引号
            parts = line.split()
            if len(parts) >= 2:
                space_group = parts[1].strip("'\"")
                cell_params['space_group'] = space_group

        # 进入原子块
        elif line.startswith('loop_'):
            in_atom_block = False
            atom_labels = []

        # 原子标签
        elif line.startswith('_atom_site_label'):
            in_atom_block = True
            atom_labels.append('label')
        elif line.startswith('_atom_site_type_symbol'):
            atom_labels.append('symbol')
        elif line.startswith('_atom_site_fract_x'):
            atom_labels.append('x')
        elif line.startswith('_atom_site_fract_y'):
            atom_labels.append('y')
        elif line.startswith('_atom_site_fract_z'):
            atom_labels.append('z')
        elif line.startswith('_atom_site_occupancy'):
            atom_labels.append('occupancy')

        # 解析原子数据
        elif in_atom_block and line and not line.startswith('_') and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= len(atom_labels):
                atom = {}
                for i, label in enumerate(atom_labels):
                    if label == 'x' or label == 'y' or label == 'z' or label == 'occupancy':
                        atom[label] = float(parts[i])
                    else:
                        atom[label] = parts[i]
                atoms.append(atom)

    return cell_params, atoms, lines


def generate_cif_content(cell_params: Dict, atoms: List[Dict],
                         symops: List[str], original_lines: List[str]) -> str:
    """
    生成完整的CIF文件内容

    Args:
        cell_params: 晶胞参数字典
        atoms: 原子列表
        symops: 对称操作列表
        original_lines: 原始文件行

    Returns:
        CIF文件内容字符串
    """
    # 创建输出内容
    output = []

    # 添加晶胞参数
    output.append(f"_cell_length_a {cell_params.get('a', 0):.6f}")
    output.append(f"_cell_length_b {cell_params.get('b', 0):.6f}")
    output.append(f"_cell_length_c {cell_params.get('c', 0):.6f}")
    output.append(f"_cell_angle_alpha {cell_params.get('alpha', 90):.6f}")
    output.append(f"_cell_angle_beta {cell_params.get('beta', 90):.6f}")
    output.append(f"_cell_angle_gamma {cell_params.get('gamma', 90):.6f}")
    output.append("")

    # 添加空间群信息
    space_group = cell_params.get('space_group', 'P1')
    output.append(f"_symmetry_space_group_name_H-M '{space_group}'")

    # 添加对称操作
    output.append("loop_")
    output.append("_symmetry_equiv_pos_site_id")
    output.append("_symmetry_equiv_pos_as_xyz")
    for i, symop in enumerate(symops, 1):
        output.append(f"{i} '{symop}'")
    output.append("")

    # 添加原子信息
    output.append("loop_")
    output.append("_atom_site_label")
    output.append("_atom_site_type_symbol")
    output.append("_atom_site_fract_x")
    output.append("_atom_site_fract_y")
    output.append("_atom_site_fract_z")
    output.append("_atom_site_occupancy")

    for atom in atoms:
        output.append(f"{atom['label']:4s} {atom['symbol']:2s} "
                      f"{atom['x']:.8f} {atom['y']:.8f} {atom['z']:.8f} "
                      f"{atom['occupancy']:.6f}")

    # 添加原始文件中的额外信息（注释等）
    output.append("")
    for line in original_lines:
        if line.strip().startswith('#') and 'BindingEnergy' in line:
            output.append(line.rstrip())
        elif line.strip().startswith('#') and 'FitnessScore' in line:
            output.append(line.rstrip())
        elif line.strip().startswith('#') and 'NumAtoms' in line:
            # 更新原子数
            output.append(f"#NumAtoms: {len(atoms)}")
        elif line.strip().startswith('#') and 'CrystalSystem' in line:
            # 根据空间群确定晶系
            space_group = cell_params.get('space_group', '').upper()
            if space_group.startswith('P1') or space_group.startswith('P-1'):
                crystal_system = 'triclinic'
            elif space_group.startswith('P2') or space_group.startswith('C2') or space_group.startswith('I2'):
                crystal_system = 'monoclinic'
            elif space_group.startswith('P222') or space_group.startswith('C222') or space_group.startswith(
                    'F222') or space_group.startswith('I222'):
                crystal_system = 'orthorhombic'
            elif space_group.startswith('P4') or space_group.startswith('I4'):
                crystal_system = 'tetragonal'
            elif space_group.startswith('R') or space_group.startswith('P3') or space_group.startswith('H3'):
                crystal_system = 'trigonal'
            elif space_group.startswith('P6') or space_group.startswith('H6'):
                crystal_system = 'hexagonal'
            elif space_group.startswith('P23') or space_group.startswith('F23') or space_group.startswith('I23'):
                crystal_system = 'cubic'
            else:
                crystal_system = 'unknown'
            output.append(f"#CrystalSystem: {crystal_system}")

    return '\n'.join(output)


def process_cif_file(input_file: str, expander: CIFSymmetryExpander) -> bool:
    """
    处理单个CIF文件

    Args:
        input_file: 输入CIF文件路径
        expander: CIF对称性扩展器

    Returns:
        处理成功返回True，否则返回False
    """
    try:
        print(f"处理文件: {input_file}")

        # 解析CIF文件
        cell_params, atoms, original_lines = parse_cif_file(input_file)

        if not cell_params or not atoms:
            print(f"  警告: 文件 {input_file} 没有有效的晶胞参数或原子数据")
            return False

        # 获取空间群
        space_group = cell_params.get('space_group', 'P1')
        print(f"  空间群: {space_group}")
        print(f"  原始原子数: {len(atoms)}")

        # 扩展原子
        expanded_atoms = expander.expand_atoms(atoms, space_group)
        print(f"  扩展后原子数: {len(expanded_atoms)}")

        # 获取对称操作
        symops = expander.get_symmetry_operations(space_group)

        # 生成新的CIF内容
        cif_content = generate_cif_content(cell_params, expanded_atoms, symops, original_lines)

        # 保存文件（覆盖原文件）
        with open(input_file, 'w') as f:
            f.write(cif_content)

        print(f"  文件已更新")
        return True

    except Exception as e:
        print(f"  处理文件 {input_file} 时出错: {e}")
        return False


def main():
    """
    主函数：处理指定目录下的所有CIF文件
    """
    import argparse

    parser = argparse.ArgumentParser(description='将CIF文件中的独立原子扩展为完整的晶体结构')
    parser.add_argument('directory', help='包含CIF文件的目录路径')
    parser.add_argument('--tolerance', type=float, default=0.01,
                        help='原子位置去重容差 (默认: 0.01)')
    parser.add_argument('--extension', default='.cif',
                        help='CIF文件扩展名 (默认: .cif)')

    args = parser.parse_args()

    # 创建对称性扩展器
    expander = CIFSymmetryExpander(tolerance=args.tolerance)

    # 获取目录中的所有CIF文件
    directory = Path(args.directory)
    if not directory.exists():
        print(f"错误: 目录 {directory} 不存在")
        return

    cif_files = list(directory.glob(f"*{args.extension}"))
    if not cif_files:
        print(f"警告: 在目录 {directory} 中没有找到 {args.extension} 文件")
        return

    print(f"找到 {len(cif_files)} 个CIF文件")
    print("=" * 50)

    # 处理每个文件
    successful = 0
    for cif_file in cif_files:
        if process_cif_file(str(cif_file), expander):
            successful += 1
        print("-" * 50)

    print(f"处理完成: {successful}/{len(cif_files)} 个文件成功处理")


if __name__ == "__main__":
    main()