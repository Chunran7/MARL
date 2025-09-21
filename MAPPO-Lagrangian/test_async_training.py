#!/usr/bin/env python
"""
异步MAPPO-Lagrangian训练测试脚本
用于验证不同异步策略的效果
"""

import sys
import os
import numpy as np
import torch
import time
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_async_training():
    """测试异步训练的不同策略"""
    
    print("=" * 60)
    print("异步MAPPO-Lagrangian训练测试")
    print("=" * 60)
    
    # 测试配置
    test_configs = [
        {
            'name': '分层异步更新',
            'async_mode': 'hierarchical',
            'selection_ratio': 0.5,
            'hierarchy_levels': 3
        },
        {
            'name': '重要性采样异步更新', 
            'async_mode': 'importance',
            'selection_ratio': 0.6,
            'importance_threshold': 0.1
        },
        {
            'name': '渐进式异步更新',
            'async_mode': 'progressive',
            'min_selection_ratio': 0.3,
            'max_selection_ratio': 1.0
        }
    ]
    
    # 环境配置
    env_configs = [
        {
            'env_name': 'HalfCheetah-v2',
            'num_agents': 6,
            'episode_length': 1000
        },
        {
            'env_name': 'Ant-v2', 
            'num_agents': 8,
            'episode_length': 1000
        }
    ]
    
    results = {}
    
    for env_config in env_configs:
        env_name = env_config['env_name']
        print(f"\n测试环境: {env_name}")
        print("-" * 40)
        
        results[env_name] = {}
        
        for config in test_configs:
            config_name = config['name']
            print(f"\n  策略: {config_name}")
            
            # 模拟训练过程
            start_time = time.time()
            
            # 这里应该调用实际的训练函数
            # 为了演示，我们模拟一些结果
            training_result = simulate_training(env_config, config)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            results[env_name][config_name] = {
                'training_time': training_time,
                'final_reward': training_result['final_reward'],
                'final_cost': training_result['final_cost'],
                'communication_efficiency': training_result['communication_efficiency'],
                'convergence_episodes': training_result['convergence_episodes']
            }
            
            print(f"    训练时间: {training_time:.2f}s")
            print(f"    最终奖励: {training_result['final_reward']:.2f}")
            print(f"    最终成本: {training_result['final_cost']:.2f}")
            print(f"    通信效率: {training_result['communication_efficiency']:.2f}")
            print(f"    收敛回合: {training_result['convergence_episodes']}")
    
    # 生成对比报告
    generate_comparison_report(results)

def simulate_training(env_config, async_config):
    """
    模拟训练过程（实际使用时应该调用真实的训练函数）
    """
    num_agents = env_config['num_agents']
    async_mode = async_config['async_mode']
    
    # 模拟不同策略的效果
    if async_mode == 'hierarchical':
        # 分层策略通常有较好的稳定性
        final_reward = np.random.normal(2500, 200)
        final_cost = np.random.normal(15, 3)
        communication_efficiency = async_config['selection_ratio']
        convergence_episodes = np.random.randint(800, 1200)
        
    elif async_mode == 'importance':
        # 重要性采样策略效果较好但可能不稳定
        final_reward = np.random.normal(2700, 300)
        final_cost = np.random.normal(12, 4)
        communication_efficiency = async_config['selection_ratio'] * 0.9
        convergence_episodes = np.random.randint(600, 1000)
        
    elif async_mode == 'progressive':
        # 渐进式策略平衡效果和效率
        final_reward = np.random.normal(2600, 250)
        final_cost = np.random.normal(13, 3)
        communication_efficiency = (async_config['min_selection_ratio'] + 
                                   async_config['max_selection_ratio']) / 2
        convergence_episodes = np.random.randint(700, 1100)
    
    return {
        'final_reward': final_reward,
        'final_cost': final_cost,
        'communication_efficiency': communication_efficiency,
        'convergence_episodes': convergence_episodes
    }

def generate_comparison_report(results):
    """生成对比报告"""
    print("\n" + "=" * 60)
    print("异步训练策略对比报告")
    print("=" * 60)
    
    for env_name, env_results in results.items():
        print(f"\n环境: {env_name}")
        print("-" * 40)
        
        # 创建对比表格
        strategies = list(env_results.keys())
        metrics = ['final_reward', 'final_cost', 'communication_efficiency', 'convergence_episodes']
        
        print(f"{'策略':<15} {'最终奖励':<10} {'最终成本':<10} {'通信效率':<10} {'收敛回合':<10}")
        print("-" * 60)
        
        for strategy in strategies:
            result = env_results[strategy]
            print(f"{strategy:<15} {result['final_reward']:<10.1f} {result['final_cost']:<10.1f} "
                  f"{result['communication_efficiency']:<10.2f} {result['convergence_episodes']:<10}")
        
        # 找出最佳策略
        best_reward_strategy = max(strategies, key=lambda x: env_results[x]['final_reward'])
        best_efficiency_strategy = max(strategies, key=lambda x: env_results[x]['communication_efficiency'])
        
        print(f"\n最佳奖励策略: {best_reward_strategy}")
        print(f"最高效率策略: {best_efficiency_strategy}")

def run_actual_training_test():
    """运行实际的训练测试（需要完整环境）"""
    try:
        # 导入训练脚本
        from train_mujoco_async import main as async_main
        
        print("运行实际异步训练测试...")
        
        # 测试参数
        test_args = [
            '--env_name', 'HalfCheetah-v2',
            '--algorithm_name', 'rmappo_lagr',
            '--experiment_name', 'async_test',
            '--num_agents', '6',
            '--num_env_steps', '10000',  # 较短的测试
            '--async_mode', 'hierarchical',
            '--selection_ratio', '0.5',
            '--use_wandb', 'False'
        ]
        
        # 模拟命令行参数
        import sys
        original_argv = sys.argv
        sys.argv = ['test_async_training.py'] + test_args
        
        try:
            async_main()
        finally:
            sys.argv = original_argv
            
    except ImportError as e:
        print(f"无法导入训练模块: {e}")
        print("请确保所有依赖都已正确安装")
    except Exception as e:
        print(f"训练测试出错: {e}")

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 模拟测试（快速）")
    print("2. 实际训练测试（需要完整环境）")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        test_async_training()
    elif choice == "2":
        run_actual_training_test()
    else:
        print("无效选择，运行模拟测试...")
        test_async_training()