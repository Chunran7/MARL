# 异步MAPPO-Lagrangian配置包

# 导入异步配置
from .async_config import AsyncConfig, HIERARCHICAL_CONFIG, IMPORTANCE_CONFIG, PROGRESSIVE_CONFIG

# 导入原始的get_config函数
import sys
import os

# 获取mappo_lagrangian目录的路径
mappo_lagrangian_dir = os.path.dirname(os.path.dirname(__file__))
config_py_path = os.path.join(mappo_lagrangian_dir, 'config.py')

# 动态导入config.py模块
import importlib.util
spec = importlib.util.spec_from_file_location("original_config", config_py_path)
original_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original_config)

# 获取get_config函数
get_config = original_config.get_config

__all__ = [
    'AsyncConfig',
    'HIERARCHICAL_CONFIG', 
    'IMPORTANCE_CONFIG',
    'PROGRESSIVE_CONFIG',
    'get_config'
]