import importlib.util
import sys
from pathlib import Path

def import_file(module_name: str, file_path: str, make_importable: bool = False):
    """
    从指定文件路径动态导入模块（Python 3.8+ 版本）。

    Args:
        module_name (str): 模块名称（可以是任意自定义名称）
        file_path (str): 目标 Python 文件路径
        make_importable (bool): 是否将模块注册到 sys.modules 方便后续 import

    Returns:
        module: 导入的模块对象
    """
    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"模块文件不存在: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为模块创建 spec: {module_name} ({file_path})")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if make_importable:
        sys.modules[module_name] = module

    return module
