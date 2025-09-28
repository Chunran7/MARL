import time
from itertools import chain

import wandb
import numpy as np
from functools import reduce
import torch
from mappo_lagrangian.runner.separated.async_base_runner_mappo_lagr import AsyncRunner


def _t2n(x):
    return x.detach().cpu().numpy()


class AsyncMujocoRunner(AsyncRunner):
    """
    异步MAPPO-Lagrangian的Mujoco训练器
    继承自AsyncRunner，实现Mujoco环境特定的功能
    """

    def __init__(self, config):
        super(AsyncMujocoRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        train_episode_costs = [0 for _ in range(self.n_rollout_threads)]

        for episode in range(episodes):
            # 更新当前episode用于渐进式异步
            self.current_episode = episode
            self.total_episodes = episodes
            
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            done_episodes_rewards = []
            done_episodes_costs = []

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, \
                rnn_states_cost, actions_env = self.collect(step)

                # Observe reward cost and next obs
                obs, share_obs, rewards, costs, dones, infos, _ = self.envs.step(actions_env)

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                cost_env = np.mean(costs, axis=1).flatten()
                train_episode_rewards += reward_env
                train_episode_costs += cost_env
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        done_episodes_costs.append(train_episode_costs[t])
                        train_episode_rewards[t] = 0
                        train_episode_costs[t] = 0

                data = obs, share_obs, rewards, costs, dones, infos, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic, cost_preds, rnn_states_cost

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            
            # 使用异步训练方法
            train_infos, cost_train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    aver_episode_costs = np.mean(done_episodes_costs)
                    print("some episodes done, average episode reward is {}, average episode cost is {}".format(
                        aver_episode_rewards, aver_episode_costs))
                    self.log_train(train_infos, total_num_steps)
                    self.log_env({"aver_episode_rewards": aver_episode_rewards}, total_num_steps)
                    self.log_env({"aver_episode_costs": aver_episode_costs}, total_num_steps)
                    
                    # 记录异步训练特定的指标
                    if hasattr(self, 'agent_importance_history') and len(self.agent_importance_history) > 0:
                        avg_importance = np.mean(self.agent_importance_history[-1])
                        self.log_env({"avg_agent_importance": avg_importance}, total_num_steps)
                    
                    # 记录异步模式信息
                    self.log_env({"async_mode": self.async_mode}, total_num_steps)
                    self.log_env({"async_ratio": self.async_ratio}, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def return_aver_cost(self, aver_episode_costs):
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].return_aver_cost(aver_episode_costs)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []
        cost_preds = []
        rnn_states_cost = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic, cost_pred, rnn_state_cost \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            rnn_states_cost=self.buffer[agent_id].rnn_states_cost[step])
            values.append(_t2n(value))
            action = _t2n(action)
            actions.append(action)
            temp_actions_env.append(action[0])
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))
            cost_preds.append(_t2n(cost_pred))
            rnn_states_cost.append(_t2n(rnn_state_cost))

        # [agents, envs, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                # 确保索引不越界
                if i < len(temp_action_env):
                    one_hot_action_env.append(temp_action_env[i])
                else:
                    # 如果索引越界，使用第一个环境的动作作为默认值
                    one_hot_action_env.append(temp_action_env[0])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)
        cost_preds = np.array(cost_preds).transpose(1, 0, 2)
        rnn_states_cost = np.array(rnn_states_cost).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost, actions_env

    def insert(self, data):
        obs, share_obs, rewards, costs, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]),
            dtype=np.float32)
        rnn_states_cost[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_cost.shape[2:]),
            dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id], rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id], actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id], rewards[:, agent_id], masks[:, agent_id], bad_masks[:, agent_id],
                                         active_masks[:, agent_id], None, costs[:, agent_id], 
                                         cost_preds[:, agent_id], rnn_states_cost[:, agent_id])

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    # 确保v是数值类型，避免workspace错误
                    if isinstance(v, (list, np.ndarray)):
                        v = float(np.mean(v))
                    elif isinstance(v, str):
                        continue
                    else:
                        v = float(v)
                    self.writter.add_scalar(agent_k, v, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_episode_costs = []
        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(eval_obs[:, agent_id],
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)
                eval_action = _t2n(eval_action)
                eval_temp_actions_env.append(eval_action[0])
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    # 确保索引不越界
                    if i < len(eval_temp_action_env):
                        eval_one_hot_action_env.append(eval_temp_action_env[i])
                    else:
                        # 如果索引越界，使用第一个环境的动作作为默认值
                        eval_one_hot_action_env.append(eval_temp_action_env[0])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_costs, eval_dones, eval_infos, _ = self.eval_envs.step(
                eval_actions_env)
            eval_episode_rewards.append(eval_rewards)
            eval_episode_costs.append(eval_costs)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_episode_costs = np.array(eval_episode_costs)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_average_episode_costs = np.mean(np.sum(eval_episode_costs[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards,
                                     'eval_average_episode_costs': eval_average_episode_costs})

        self.log_env(eval_train_infos, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        if isinstance(env_infos, dict):
            # 单个字典的情况
            for k, v in env_infos.items():
                if self.use_wandb:
                    wandb.log({k: v}, step=total_num_steps)
                else:
                    # 确保v是数值类型，避免workspace错误
                    if isinstance(v, (list, np.ndarray)):
                        v = float(np.mean(v))
                    elif isinstance(v, str):
                        # 如果是字符串，跳过记录或转换为数值
                        continue
                    else:
                        v = float(v)
                    self.writter.add_scalar(k, v, total_num_steps)
        else:
            # 列表的情况（原始逻辑）
            for agent_id in range(self.num_agents):
                for k, v in env_infos[agent_id].items():
                    agent_k = "agent%i/" % agent_id + k
                    if self.use_wandb:
                        wandb.log({agent_k: v}, step=total_num_steps)
                    else:
                        # 确保v是数值类型
                        if isinstance(v, (list, np.ndarray)):
                            v = float(np.mean(v))
                        elif isinstance(v, str):
                            continue
                        else:
                            v = float(v)
                        self.writter.add_scalar(agent_k, v, total_num_steps)