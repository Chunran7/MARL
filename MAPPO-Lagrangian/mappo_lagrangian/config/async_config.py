"""
异步MAPPO-Lagrangian训练配置
"""

class AsyncConfig:
    def __init__(self):
        # 异步训练模式选择
        # 'hierarchical': 分层异步 - 将智能体分组，组内同步，组间异步
        # 'importance': 重要性采样 - 基于智能体重要性选择性更新
        # 'progressive': 渐进式异步 - 根据训练进度调整同步程度
        self.async_mode = 'hierarchical'
        
        # 异步更新比例 (0.0-1.0)
        # 对于importance模式：每次更新选择的智能体比例
        # 对于hierarchical模式：影响分组策略
        self.async_ratio = 0.7
        
        # 分层模式参数
        self.num_groups = 4  # 智能体分组数量
        
        # 重要性采样参数
        self.importance_window = 10  # 重要性计算的历史窗口大小
        self.importance_update_freq = 5  # 重要性分数更新频率
        
        # 渐进式异步参数
        self.initial_sync_ratio = 1.0  # 初始同步比例
        self.final_sync_ratio = 0.3    # 最终同步比例
        self.sync_decay_episodes = 1000  # 同步比例衰减的episode数
        
        # 轻量级更新参数
        self.value_update_freq = 2  # 价值函数更新频率（相对于策略更新）
        self.lightweight_update_ratio = 0.5  # 轻量级更新的学习率衰减
        
        # 性能优化参数
        self.enable_gradient_accumulation = True  # 启用梯度累积
        self.gradient_accumulation_steps = 2     # 梯度累积步数
        self.enable_mixed_precision = False      # 启用混合精度训练
        
        # 通信优化参数
        self.enable_communication_compression = True  # 启用通信压缩
        self.compression_ratio = 0.1  # 压缩比例
        
        # 稳定性参数
        self.stability_check_freq = 50  # 稳定性检查频率
        self.max_policy_divergence = 0.5  # 最大策略分歧阈值
        self.sync_threshold = 0.8  # 强制同步阈值
        
    def get_config_dict(self):
        """
        返回配置字典
        """
        return {
            'async_mode': self.async_mode,
            'async_ratio': self.async_ratio,
            'num_groups': self.num_groups,
            'importance_window': self.importance_window,
            'importance_update_freq': self.importance_update_freq,
            'initial_sync_ratio': self.initial_sync_ratio,
            'final_sync_ratio': self.final_sync_ratio,
            'sync_decay_episodes': self.sync_decay_episodes,
            'value_update_freq': self.value_update_freq,
            'lightweight_update_ratio': self.lightweight_update_ratio,
            'enable_gradient_accumulation': self.enable_gradient_accumulation,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'enable_mixed_precision': self.enable_mixed_precision,
            'enable_communication_compression': self.enable_communication_compression,
            'compression_ratio': self.compression_ratio,
            'stability_check_freq': self.stability_check_freq,
            'max_policy_divergence': self.max_policy_divergence,
            'sync_threshold': self.sync_threshold,
        }
    
    def update_from_args(self, args):
        """
        从命令行参数更新配置
        """
        if hasattr(args, 'async_mode'):
            self.async_mode = args.async_mode
        if hasattr(args, 'async_ratio'):
            self.async_ratio = args.async_ratio
        if hasattr(args, 'num_groups'):
            self.num_groups = args.num_groups
        if hasattr(args, 'importance_window'):
            self.importance_window = args.importance_window
            
    def validate_config(self):
        """
        验证配置的有效性
        """
        assert self.async_mode in ['hierarchical', 'importance', 'progressive'], \
            f"Invalid async_mode: {self.async_mode}"
        assert 0.0 <= self.async_ratio <= 1.0, \
            f"async_ratio must be between 0.0 and 1.0, got {self.async_ratio}"
        assert self.num_groups > 0, \
            f"num_groups must be positive, got {self.num_groups}"
        assert self.importance_window > 0, \
            f"importance_window must be positive, got {self.importance_window}"
        
        return True


# 预定义配置
HIERARCHICAL_CONFIG = AsyncConfig()
HIERARCHICAL_CONFIG.async_mode = 'hierarchical'
HIERARCHICAL_CONFIG.async_ratio = 0.7
HIERARCHICAL_CONFIG.num_groups = 4

IMPORTANCE_CONFIG = AsyncConfig()
IMPORTANCE_CONFIG.async_mode = 'importance'
IMPORTANCE_CONFIG.async_ratio = 0.6
IMPORTANCE_CONFIG.importance_window = 15

PROGRESSIVE_CONFIG = AsyncConfig()
PROGRESSIVE_CONFIG.async_mode = 'progressive'
PROGRESSIVE_CONFIG.async_ratio = 0.7  # 设置为与默认值一致
PROGRESSIVE_CONFIG.initial_sync_ratio = 1.0
PROGRESSIVE_CONFIG.final_sync_ratio = 0.3
PROGRESSIVE_CONFIG.sync_decay_episodes = 800