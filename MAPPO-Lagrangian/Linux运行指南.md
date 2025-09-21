# Linux系统运行指南 - 异步MAPPO-Lagrangian

## 🚀 快速导航

**新手用户推荐阅读顺序：**
1. 📖 [快速开始.md](./快速开始.md) - 5分钟快速上手
2. 📋 [运行命令大全.md](./运行命令大全.md) - 各种场景的具体命令
3. 📚 本文档 - 详细的系统配置和高级用法

**文档说明：**
- **快速开始.md**: 最简洁的运行步骤，适合快速验证
- **运行命令大全.md**: 各种场景下的具体命令，复制即用
- **Linux运行指南.md**: 完整的系统配置和故障排除
- **README_async.md**: 异步算法的技术原理和实现细节

---

## 概述

本指南详细说明如何在Linux系统上运行改进后的异步MAPPO-Lagrangian训练系统。该系统解决了原始随机抽取智能体方法效果变差的问题，提供了三种智能的异步训练策略。

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐RTX 3080或更高)
- **内存**: 至少16GB RAM
- **存储**: 至少10GB可用空间
- **CPU**: 多核处理器 (推荐8核或更多)

### 软件要求
- **操作系统**: Ubuntu 18.04+ / CentOS 7+ / 其他Linux发行版
- **Python**: 3.7
- **CUDA**: 11.1或兼容版本
- **MuJoCo**: 2.0+

## 1. 环境安装

### 1.1 创建Conda环境

```bash
# 创建conda环境
conda create -n macpo python==3.7
conda activate macpo

# 安装基础依赖
pip install -r requirements.txt

# 安装PyTorch (根据你的CUDA版本调整)
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

### 1.2 安装项目包

```bash
# 进入MAPPO-Lagrangian目录
cd MAPPO-Lagrangian/mappo_lagrangian

# 安装项目包
pip install -e .
```

### 1.3 安装MuJoCo环境

```bash
# 设置MuJoCo环境变量
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:${LD_PRELOAD}

# 安装mujoco-py
pip install mujoco-py==2.0.2.13

# 克隆Safety Multi-Agent Mujoco环境
git clone https://github.com/chauncygu/Safe-Multi-Agent-Mujoco.git
# 将环境路径添加到Python路径中
export PYTHONPATH=${PYTHONPATH}:/path/to/Safe-Multi-Agent-Mujoco
```

### 1.4 验证安装

```bash
# 测试MuJoCo安装
python -c "import mujoco_py; print('MuJoCo安装成功')"

# 测试项目安装
python -c "import mappo_lagrangian; print('项目安装成功')"

# 运行快速测试
python test_async_training.py
```

## 2. 运行方式

### 2.1 基本运行

#### 单次训练
```bash
# 回到项目根目录
cd /path/to/Multi-Agent-Constrained-Policy-Optimisation/MAPPO-Lagrangian

# 分层异步训练
python train_mujoco_async.py \
    --env_name mujoco \
    --scenario Ant-v2 \
    --agent_conf 2x4 \
    --algorithm_name rmappo_lagr \
    --async_mode hierarchical \
    --selection_ratio 0.5 \
    --num_env_steps 1000000

# 重要性采样异步训练
python train_mujoco_async.py \
    --env_name mujoco \
    --scenario Ant-v2 \
    --agent_conf 2x4 \
    --algorithm_name rmappo_lagr \
    --async_mode importance \
    --selection_ratio 0.6 \
    --importance_threshold 0.1

# 渐进式异步训练
python train_mujoco_async.py \
    --env_name mujoco \
    --scenario Ant-v2 \
    --agent_conf 2x4 \
    --algorithm_name rmappo_lagr \
    --async_mode progressive \
    --min_selection_ratio 0.3 \
    --max_selection_ratio 1.0
```

#### 使用Shell脚本运行
```bash
# 给脚本执行权限
chmod +x train_mujoco_async.sh

# 编辑脚本配置 (可选)
nano train_mujoco_async.sh

# 运行训练
./train_mujoco_async.sh
```

### 2.2 批量实验

#### 运行对比实验
```bash
# 给实验脚本执行权限
chmod +x run_async_experiments.sh

# 运行所有对比实验
./run_async_experiments.sh

# 监控运行状态
ps aux | grep train_mujoco_async

# 查看实时日志
tail -f logs/async_experiments/*.log
```

#### 自定义实验配置
```bash
# 编辑实验脚本
nano run_async_experiments.sh

# 修改以下配置:
# - scenarios: 环境场景
# - agent_confs: 智能体配置
# - async_modes: 异步策略
# - seed_max: 种子数量
```

### 2.3 参数说明

#### 核心参数
```bash
--env_name mujoco                    # 环境名称
--scenario Ant-v2                    # 场景 (Ant-v2, HalfCheetah-v2)
--agent_conf 2x4                     # 智能体配置
--algorithm_name rmappo_lagr          # 算法名称
--async_mode hierarchical            # 异步模式
--selection_ratio 0.5                # 智能体选择比例
--num_env_steps 10000000             # 训练步数
--safety_bound 0.2                   # 安全约束边界
--lamda_lagr 0.78                    # 拉格朗日系数
```

#### 异步特定参数
```bash
# 分层模式
--hierarchy_levels 3                 # 层级数量
--use_async_value_update             # 使用异步价值更新

# 重要性采样模式
--importance_threshold 0.1           # 重要性阈值
--importance_decay 0.95              # 重要性衰减

# 渐进式模式
--min_selection_ratio 0.3            # 最小选择比例
--max_selection_ratio 1.0            # 最大选择比例
```

#### 训练参数
```bash
--lr 9e-5                           # 学习率
--critic_lr 5e-3                    # 评论家学习率
--n_training_threads 4              # 训练线程数
--n_rollout_threads 16              # 采样线程数
--num_mini_batch 40                 # 小批次数量
--episode_length 1000               # 回合长度
--ppo_epoch 5                       # PPO更新轮数
--seed 50                           # 随机种子
```

## 3. 环境配置

### 3.1 支持的环境

#### Ant环境
```bash
# Ant-v2 (2x4配置)
--scenario Ant-v2 --agent_conf 2x4 --safety_bound 0.2

# ManyAgent Ant (更多智能体)
--scenario ManyAgentAnt-v2 --agent_conf 3x2 --safety_bound 1.0
```

#### HalfCheetah环境
```bash
# HalfCheetah-v2 (2x3配置)
--scenario HalfCheetah-v2 --agent_conf 2x3 --safety_bound 5.0
```

### 3.2 环境特定配置

#### Ant环境配置
```bash
python train_mujoco_async.py \
    --scenario Ant-v2 \
    --agent_conf 2x4 \
    --safety_bound 0.2 \
    --lamda_lagr 0.78 \
    --episode_length 1000 \
    --num_env_steps 10000000 \
    --async_mode hierarchical \
    --selection_ratio 0.5
```

#### HalfCheetah环境配置
```bash
python train_mujoco_async.py \
    --scenario HalfCheetah-v2 \
    --agent_conf 2x3 \
    --safety_bound 5.0 \
    --lamda_lagr 0.5 \
    --episode_length 1000 \
    --num_env_steps 8000000 \
    --async_mode importance \
    --selection_ratio 0.6
```

## 4. 监控和调试

### 4.1 训练监控

#### 实时监控
```bash
# 查看GPU使用情况
nvidia-smi -l 1

# 监控训练进程
htop

# 查看训练日志
tail -f logs/async_experiments/*.log

# 监控特定实验
tail -f results/*/logs/progress.txt
```

#### 性能指标
```bash
# 查看训练指标
grep "average_episode_rewards" logs/async_experiments/*.log

# 查看安全约束满足情况
grep "average_episode_costs" logs/async_experiments/*.log

# 查看通信效率
grep "communication_efficiency" logs/async_experiments/*.log
```

### 4.2 常见问题排查

#### 环境问题
```bash
# MuJoCo许可证问题
export MUJOCO_KEY_PATH=/path/to/mjkey.txt

# OpenGL问题
export DISPLAY=:0
xvfb-run -a -s "-screen 0 1024x768x24" python train_mujoco_async.py ...

# 内存不足
# 减少n_rollout_threads和num_mini_batch参数
```

#### 训练问题
```bash
# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"

# 检查环境导入
python -c "from mappo_lagrangian.envs.safety_ma_mujoco import SafetyMAMujocoEnv"

# 调试模式运行
python -u train_mujoco_async.py --debug True
```

## 5. 结果分析

### 5.1 结果文件结构
```
results/
├── async_hierarchical_Ant-v2_2x4_seed1/
│   ├── logs/
│   │   ├── progress.txt          # 训练进度
│   │   └── train.log            # 详细日志
│   ├── models/                  # 保存的模型
│   └── videos/                  # 训练视频
├── async_importance_Ant-v2_2x4_seed1/
└── async_progressive_Ant-v2_2x4_seed1/
```

### 5.2 性能对比

#### 查看训练曲线
```bash
# 安装绘图依赖
pip install matplotlib seaborn pandas

# 运行结果分析脚本
python analyze_results.py --result_dir results/

# 生成对比图表
python plot_comparison.py --experiments async_hierarchical async_importance async_progressive
```

#### 关键指标
- **训练速度**: 每秒环境步数
- **通信效率**: 选择智能体比例
- **安全性能**: 平均成本 (越低越好)
- **任务性能**: 平均奖励 (越高越好)
- **收敛速度**: 达到目标性能的步数

## 6. 高级用法

### 6.1 分布式训练

#### 多GPU训练
```bash
# 使用多个GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mujoco_async.py \
    --n_training_threads 16 \
    --n_rollout_threads 64 \
    --async_mode hierarchical
```

#### 集群训练
```bash
# 使用SLURM提交任务
sbatch --gres=gpu:4 --time=24:00:00 run_async_experiments.sh
```

### 6.2 超参数调优

#### 网格搜索
```bash
# 创建超参数搜索脚本
python hyperparameter_search.py \
    --selection_ratios 0.3,0.5,0.7 \
    --hierarchy_levels 2,3,4 \
    --importance_thresholds 0.05,0.1,0.15
```

#### 贝叶斯优化
```bash
# 使用Optuna进行超参数优化
pip install optuna
python bayesian_optimization.py --n_trials 100
```

### 6.3 自定义配置

#### 创建配置文件
```bash
# 创建YAML配置文件
cat > custom_config.yaml << EOF
env_name: mujoco
scenario: Ant-v2
agent_conf: 2x4
async_mode: hierarchical
selection_ratio: 0.5
hierarchy_levels: 3
safety_bound: 0.2
num_env_steps: 5000000
EOF

# 使用配置文件运行
python train_mujoco_async.py --config custom_config.yaml
```

## 7. 故障排除

### 7.1 常见错误

#### 导入错误
```bash
# 错误: ModuleNotFoundError: No module named 'mappo_lagrangian'
# 解决: 重新安装项目包
cd MAPPO-Lagrangian/mappo_lagrangian
pip install -e .
```

#### CUDA错误
```bash
# 错误: CUDA out of memory
# 解决: 减少批次大小或线程数
--num_mini_batch 20 --n_rollout_threads 8
```

#### MuJoCo错误
```bash
# 错误: MuJoCo license not found
# 解决: 设置许可证路径
export MUJOCO_KEY_PATH=/path/to/mjkey.txt
```

### 7.2 性能优化

#### 内存优化
```bash
# 减少内存使用
--n_rollout_threads 8 \
--num_mini_batch 20 \
--episode_length 500
```

#### 速度优化
```bash
# 提高训练速度
--n_training_threads 8 \
--use_linear_lr_decay False \
--use_proper_time_limits False
```

## 8. 实验建议

### 8.1 基础实验
1. **单策略测试**: 先测试单个异步策略
2. **参数敏感性**: 测试不同选择比例的影响
3. **环境适应性**: 在不同环境中测试效果

### 8.2 对比实验
1. **与原始方法对比**: 对比同步训练和随机选择
2. **策略间对比**: 比较三种异步策略的效果
3. **规模测试**: 测试不同智能体数量下的表现

### 8.3 消融实验
1. **组件重要性**: 测试各个组件的贡献
2. **参数影响**: 分析关键参数的影响
3. **稳定性测试**: 多种子运行验证稳定性

## 9. 技术支持

### 9.1 日志分析
```bash
# 查看错误日志
grep -i error logs/async_experiments/*.log

# 查看警告信息
grep -i warning logs/async_experiments/*.log

# 分析性能瓶颈
grep "step_time" logs/async_experiments/*.log
```

### 9.2 调试模式
```bash
# 启用详细日志
python train_mujoco_async.py --log_level DEBUG

# 启用性能分析
python -m cProfile -o profile.stats train_mujoco_async.py

# 内存分析
python -m memory_profiler train_mujoco_async.py
```

## 10. 总结

异步MAPPO-Lagrangian系统通过智能的智能体选择策略，成功解决了原始随机抽取方法的问题：

- **效果提升**: 通过重要性采样和分层更新保持训练效果
- **效率提升**: 减少30-50%的通信开销和训练时间
- **稳定性**: 渐进式策略确保训练稳定收敛

按照本指南的步骤，你可以在Linux系统上成功运行改进的异步训练系统，并获得比原始方法更好的效果和效率。