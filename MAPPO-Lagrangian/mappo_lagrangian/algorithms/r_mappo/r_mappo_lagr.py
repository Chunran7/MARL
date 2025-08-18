import numpy as np
import torch
import torch.nn as nn
from mappo_lagrangian.utils.util import get_gard_norm, huber_loss, mse_loss
from mappo_lagrangian.utils.popart import PopArt
from mappo_lagrangian.algorithms.utils.util import check

class R_MAPPO_Lagr:
    """
    Trainer class for MAPPO-L (对应论文 Algorithm 3: MAPPO-Lagrangian).
    """

    def __init__(self,
                 args,
                 policy, hvp_approach=None, attempt_feasible_recovery=False,
                 attempt_infeasible_recovery=False, revert_to_last_safe_point=False, delta_bound=0.02, safety_bound=10,
                 _backtrack_ratio=0.8, _max_backtracks=15, _constraint_name_1="trust_region",
                 _constraint_name_2="safety_region", linesearch_infeasible_recovery=True, accept_violation=False,
                 device=torch.device("cpu")):
        # 【算法3-Step 1】初始化参数 θ, λ, Critic 等
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self._damping = 0.00001  # （辅助）二阶Hessian正则项

        # PPO 相关超参数
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.gamma = args.gamma

        # 是否启用不同技巧
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        # 【算法3-初始化约束】相关参数
        self.attempt_feasible_recovery = attempt_feasible_recovery
        self.attempt_infeasible_recovery = attempt_infeasible_recovery
        self.revert_to_last_safe_point = revert_to_last_safe_point
        num_slices = 1  # （保留接口，未使用）

        # trust region & safety constraint 超参数
        self._max_quad_constraint_val = delta_bound     # trust region 边界 δ
        self._max_lin_constraint_val = safety_bound     # safety constraint 边界 c
        self._backtrack_ratio = _backtrack_ratio        # 线搜索回溯比例
        self._max_backtracks = _max_backtracks          # 最大回溯次数
        self._constraint_name_1 = _constraint_name_1    # trust region 名称
        self._constraint_name_2 = _constraint_name_2    # safety region 名称
        self._linesearch_infeasible_recovery = linesearch_infeasible_recovery
        self._accept_violation = accept_violation

        # 【算法3 λ 初始化】
        self.lagrangian_coef = args.lagrangian_coef_rate   # λ 学习率 η (公式 (29))
        self.lamda_lagr = args.lamda_lagr                 # 初始 λ (论文伪代码 line 2)
        self.safety_bound = args.safety_bound             # 安全约束 c (公式 (27))

        self._hvp_approach = hvp_approach

        # 是否启用 PopArt 正则化
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    # ---------------- Critic 损失函数 ----------------
    # 【算法3-Step 28/29】更新 reward critic 和 cost critic 时调用
    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        计算 Critic 的价值函数损失 (Reward critic 或 Cost critic 共用)
        对应论文公式 (31)
        """
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

# 代码中return_batch对应公式中的\(\hat{R}_t\)（目标回报），
# values对应\(V_{\phi}(s_t)\)（当前价值函数预测值），
# value_preds_batch对应旧的价值预测（用于裁剪操作）。当不使用裁剪（_use_clipped_value_loss为False）时，
# value_loss_original通过mse_loss(error_original)计算，即\((V_{\phi}(s_t) - \hat{R}_t)^2\)，
# 与公式（31）中的均方误差形式完全一致。当使用裁剪（_use_clipped_value_loss为True）时，
# 通过torch.max(value_loss_original, value_loss_clipped)取原始损失和裁剪损失的最大值，
# 这是对公式（31）的工程化优化，用于稳定价值网络训练，避免预测值突变。
# 代码中对损失进行均值计算（value_loss.mean()）或
# 基于掩码的加权求和（(value_loss * active_masks_batch).sum() / active_masks_batch.sum()），
# 对应公式中\(\frac{1}{B T} \sum_{b=1}^{B} \sum_{t=0}^{T}\)的批量平均操作，确保损失在样本上的整体优化。

    # ============================================================
    # 辅助函数（非算法3核心）：一阶/二阶梯度工具
    # ============================================================

    def _get_flat_grad(self, y: torch.Tensor, model: nn.Module, **kwargs) -> torch.Tensor:
        # （辅助）计算一阶梯度 ∇θ y
        grads = torch.autograd.grad(y, model.parameters(), **kwargs, allow_unused=True)
        _grads = [val for val in grads if val is not None]
        return torch.cat([grad.reshape(-1) for grad in _grads])

    def _conjugate_gradients(self, b: torch.Tensor, flat_kl_grad: torch.Tensor, nsteps: int = 10,
                             residual_tol: float = 1e-10) -> torch.Tensor:
        # （辅助）共轭梯度法求解 Hx=b，用于近似自然梯度
        x = torch.zeros_like(b)
        r, p = b.clone(), b.clone()
        rdotr = r.dot(r)
        for i in range(nsteps):
            z = self.cal_second_hessian(p, flat_kl_grad)
            alpha = rdotr / p.dot(z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            p = r + new_rdotr / rdotr * p
            rdotr = new_rdotr
        return x

    def cal_second_hessian(self, v: torch.Tensor, flat_kl_grad: torch.Tensor) -> torch.Tensor:
        # （辅助）计算 KL 的二阶 Hessian 近似
        kl_v = (flat_kl_grad * v).sum()
        flat_kl_grad_grad = self._get_flat_grad(
            kl_v, self.policy.actor, retain_graph=True).detach()
        return flat_kl_grad_grad + v * self._damping

    def _set_from_flat_params(self, model: nn.Module, flat_params: torch.Tensor) -> nn.Module:
        # （辅助）把扁平化参数向量重新赋值到模型参数
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
        return model

    # ============================================================
    # ------------------- 算法3核心：ppo_update -------------------
    # ============================================================
    def ppo_update(self, sample, update_actor=True, precomputed_eval=None,
                   precomputed_threshold=None,
                   diff_threshold=False):
        """
        【算法3 Step 4-29】
        用一个 batch 数据更新 Actor θ, Critic φ, Cost Critic φ^c, 以及 λ
        """

        # 【算法3-Step 4】采样并准备 batch 数据
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, factor_batch, cost_preds_batch, cost_returns_barch, rnn_states_cost_batch, \
        cost_adv_targ, aver_episode_costs = sample

        # 【算法3-Step 7】Evaluate policy πθ
        values, action_log_probs, dist_entropy, cost_values = self.policy.evaluate_actions(...)

        # 【公式 (25)】混合优势函数 A^λ = A - λ * A_cost
        adv_targ_hybrid = adv_targ - self.lamda_lagr * cost_adv_targ

        # 【公式 (13)(14)(24)(26)】PPO Clip surrogate objective
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)  # ρ = π/π_old
        surr1 = imp_weights * adv_targ_hybrid
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_hybrid #公式26中clip操作

        # 【算法3-Step 15】计算 L_clip^λ
        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(factor_batch * torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        policy_loss = policy_action_loss

        # 【算法3-Step 16】更新 Actor θ
        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm) \
                          if self._use_max_grad_norm else get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        # 【公式 (17)(29)】更新 λ
        delta_lamda_lagr = -((aver_episode_costs.mean() - self.safety_bound) * (1 - self.gamma)
                              + (imp_weights * cost_adv_targ)).mean().detach()
        R_Relu = torch.nn.ReLU()
        new_lamda_lagr = R_Relu(self.lamda_lagr - (delta_lamda_lagr * self.lagrangian_coef))#公式29
        self.lamda_lagr = new_lamda_lagr

        # 【算法3-Step 28】更新 Reward Critic
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm) \
                           if self._use_max_grad_norm else get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        # 【算法3-Step 29】更新 Cost Critic
        cost_loss = self.cal_value_loss(cost_values, cost_preds_batch, cost_returns_barch, active_masks_batch)
        self.policy.cost_optimizer.zero_grad()
        (cost_loss * self.value_loss_coef).backward()
        cost_grad_norm = nn.utils.clip_grad_norm_(self.policy.cost_critic.parameters(), self.max_grad_norm) \
                         if self._use_max_grad_norm else get_gard_norm(self.policy.cost_critic.parameters())
        self.policy.cost_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, cost_loss, cost_grad_norm

    # ============================================================
    # ------------------- 算法3整体训练 train ----------------------
    # ============================================================
    def train(self, buffer, update_actor=True):
        """
        【算法3-Step 3-29】整体训练循环：
        - 计算 reward & cost advantage
        - 多次 PPO epoch 更新 θ, λ, critic
        """
        # reward advantage (GAE)
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages = (advantages - np.nanmean(advantages)) / (np.nanstd(advantages) + 1e-5)

        # cost advantage
        if self._use_popart:
            cost_adv = buffer.cost_returns[:-1] - self.value_normalizer.denormalize(buffer.cost_preds[:-1])
        else:
            cost_adv = buffer.cost_returns[:-1] - buffer.cost_preds[:-1]
        cost_adv = (cost_adv - np.nanmean(cost_adv)) / (np.nanstd(cost_adv) + 1e-5)

        train_info = {"value_loss":0,"policy_loss":0,"dist_entropy":0,"actor_grad_norm":0,
                      "critic_grad_norm":0,"ratio":0,"cost_grad_norm":0,"cost_loss":0}
        
        # 【算法3-Step 14-23】循环 PPO epoch 更新
        for _ in range(self.ppo_epoch):
            if self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch, cost_adv)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch, cost_adv=cost_adv)

            for sample in data_generator:
                # 调用 ppo_update 完成 【Step 15-29】
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, cost_loss, cost_grad_norm \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['cost_loss'] += cost_loss.item()
                train_info['cost_grad_norm'] += cost_grad_norm

        # 平均化
        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    # ---------------- 训练/采样模式切换 ----------------
    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.cost_critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.cost_critic.eval()
