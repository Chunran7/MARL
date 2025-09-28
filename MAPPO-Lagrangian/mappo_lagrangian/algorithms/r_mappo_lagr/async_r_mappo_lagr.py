import numpy as np
import torch
import torch.nn as nn
from mappo_lagrangian.algorithms.r_mappo.r_mappo_lagr import R_MAPPO_Lagr
from mappo_lagrangian.utils.util import get_gard_norm, huber_loss, mse_loss
from mappo_lagrangian.algorithms.utils.util import check


class Async_R_MAPPO_Lagr(R_MAPPO_Lagr):
    """
    异步MAPPO-Lagrangian算法实现
    支持选择性智能体更新和轻量级价值函数更新
    """
    
    def __init__(self, args, policy, device=torch.device("cpu")):
        super(Async_R_MAPPO_Lagr, self).__init__(args, policy, device)
        
        # 异步训练相关参数
        self.async_update_ratio = getattr(args, 'async_update_ratio', 0.7)
        self.value_update_freq = getattr(args, 'value_update_freq', 2)
        
        # 修复缺失的cost_value_normalizer属性（实际上应该使用value_normalizer）
        # 在标准版本中，cost相关的值也是使用value_normalizer来处理的
        self.cost_value_normalizer = self.value_normalizer
        
        # 用于跟踪梯度信息
        self.gradient_history = []
        self.max_gradient_history = 100
        
    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        计算价值损失，支持异步更新时的稳定性改进
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm:
            # PopArt的normalize方法会自动更新统计信息，不需要单独调用update
            normalized_return = self.value_normalizer(return_batch)
            error_clipped = normalized_return - value_pred_clipped
            error_original = normalized_return - values
        else:
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
    
    def cal_cost_value_loss(self, cost_values, cost_preds_batch, cost_return_batch, active_masks_batch):
        """
        计算成本价值损失，支持异步更新
        """
        cost_pred_clipped = cost_preds_batch + (cost_values - cost_preds_batch).clamp(-self.clip_param,
                                                                                      self.clip_param)
        if self._use_popart or self._use_valuenorm:
            # PopArt的normalize方法会自动更新统计信息，不需要单独调用update
            normalized_cost_return = self.cost_value_normalizer(cost_return_batch)
            cost_error_clipped = normalized_cost_return - cost_pred_clipped
            cost_error_original = normalized_cost_return - cost_values
        else:
            cost_error_clipped = cost_return_batch - cost_pred_clipped
            cost_error_original = cost_return_batch - cost_values

        if self._use_huber_loss:
            cost_value_loss_clipped = huber_loss(cost_error_clipped, self.huber_delta)
            cost_value_loss_original = huber_loss(cost_error_original, self.huber_delta)
        else:
            cost_value_loss_clipped = mse_loss(cost_error_clipped)
            cost_value_loss_original = mse_loss(cost_error_original)

        if self._use_clipped_value_loss:
            cost_value_loss = torch.max(cost_value_loss_original, cost_value_loss_clipped)
        else:
            cost_value_loss = cost_value_loss_original

        if self._use_value_active_masks:
            cost_value_loss = (cost_value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            cost_value_loss = cost_value_loss.mean()

        return cost_value_loss
    
    def ppo_update(self, sample, update_actor=True):
        """
        PPO更新，支持选择性更新策略网络
        """
        # 修复参数解包问题：标准版本有18个参数，异步版本期望15个但只得到13个
        # 标准版本的sample解包：
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, factor_batch, cost_preds_batch, cost_returns_batch, rnn_states_cost_batch, \
        cost_adv_targ, aver_episode_costs = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        cost_preds_batch = check(cost_preds_batch).to(**self.tpdv)
        cost_returns_batch = check(cost_returns_batch).to(**self.tpdv)
        cost_adv_targ = check(cost_adv_targ).to(**self.tpdv)

        # 计算价值损失
        values, action_log_probs, dist_entropy, cost_values = self.policy.evaluate_actions(share_obs_batch,
                                                                                           obs_batch, 
                                                                                           rnn_states_batch, 
                                                                                           rnn_states_critic_batch, 
                                                                                           actions_batch, 
                                                                                           masks_batch,
                                                                                           available_actions_batch,
                                                                                           active_masks_batch,
                                                                                           rnn_states_cost_batch)

        # 价值函数损失
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        cost_value_loss = self.cal_cost_value_loss(cost_values, cost_preds_batch, cost_returns_batch, active_masks_batch)

        # 策略损失（仅在update_actor=True时计算）
        policy_loss = 0
        if update_actor:
            # 重要性采样比率
            imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

            # 策略梯度损失
            surr1 = imp_weights * adv_targ
            surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

            if self._use_policy_active_masks:
                policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                                dim=-1,
                                                keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
            else:
                policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

            policy_loss = policy_action_loss

            # 熵损失
            dist_entropy_loss = -self.entropy_coef * dist_entropy.mean()

            # 总策略损失
            policy_loss = policy_loss + dist_entropy_loss

        # 总损失
        if update_actor:
            loss = policy_loss + self.value_loss_coef * value_loss + self.cost_value_loss_coef * cost_value_loss
        else:
            loss = self.value_loss_coef * value_loss + self.cost_value_loss_coef * cost_value_loss

        # 反向传播
        self.policy.optimizer.zero_grad()

        if update_actor:
            (loss - self.entropy_coef * dist_entropy.mean()).backward()
        else:
            loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.optimizer.step()

        # 记录梯度信息用于重要性计算
        if update_actor:
            self.record_gradient_info(actor_grad_norm.item() if hasattr(actor_grad_norm, 'item') else actor_grad_norm)

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights
    
    def record_gradient_info(self, grad_norm):
        """
        记录梯度信息用于重要性评估
        """
        self.gradient_history.append(grad_norm)
        if len(self.gradient_history) > self.max_gradient_history:
            self.gradient_history.pop(0)
    
    def get_gradient_importance(self):
        """
        获取基于梯度的重要性分数
        """
        if len(self.gradient_history) == 0:
            return 1.0
        
        recent_grads = self.gradient_history[-10:]  # 最近10次的梯度
        return np.mean(recent_grads) if recent_grads else 1.0
    
    def train(self, buffer, update_actor=True):
        """
        训练函数，支持选择性更新
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
            cost_advantages = buffer.cost_returns[:-1] - self.cost_value_normalizer.denormalize(buffer.cost_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
            cost_advantages = buffer.cost_returns[:-1] - buffer.cost_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        cost_advantages_copy = cost_advantages.copy()
        cost_advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_cost_advantages = np.nanmean(cost_advantages_copy)
        std_cost_advantages = np.nanstd(cost_advantages_copy)
        cost_advantages = (cost_advantages - mean_cost_advantages) / (std_cost_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        # 根据是否更新策略调整训练轮数
        num_updates = self.ppo_epoch if update_actor else max(1, self.ppo_epoch // 3)

        for _ in range(num_updates):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length, cost_advantages)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch, cost_advantages)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch, cost_adv=cost_advantages)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm.item() if hasattr(actor_grad_norm, 'item') else actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm.item()
                train_info['ratio'] += imp_weights.mean().item()

        num_updates = num_updates * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info