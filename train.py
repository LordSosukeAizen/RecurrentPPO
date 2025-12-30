from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from rppo import RNNPolicy, DVN
from stockenv import StockEnv

import time

RPOPATH = "recurrent_po.pth"
VALUEPATH = "value.pth"

DEBUG = False
SLEEP_DEBUG_TIME = 20

@dataclass
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.hiddens = []

    def clear(self):
        self.__init__()


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: (T,)
    values: (T+1,)
    dones: (T,)  -- bool tensor
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0

    for t in reversed(range(T)):
        not_done = 1.0 - dones[t].float() 
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae[-1]
        
    returns = advantages + values[:-1].squeeze(-1)
    
    if DEBUG:
        print('Values: ', values[:-1].squeeze(-1).shape)
        print('Advantage: ', advantages.shape)

    
 
    return advantages, returns



def collect_rollout(env, policy_net, value_net, rollout_len, device):
    
    buffer = RolloutBuffer()

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    h = torch.zeros(
        policy_net.num_layers, 1, policy_net.hidden_dim, device=device
    )

    if DEBUG:
        print('sequence length: ', rollout_len)
    
    for _ in range(rollout_len):
        obs_seq = obs.view(1, 1, -1)  # (T=1, B=1, obs_dim)

        probs, h_next = policy_net(obs_seq, h)
        probs = probs.squeeze(0).squeeze(0)

        dist = Categorical(probs)
        action = dist.sample()

        value = value_net(h[-1]).squeeze(-1)
        
        if DEBUG:
            print('value shape: ', value.shape)

        next_obs, reward, done, _, _ = env.step(action.item())

        buffer.obs.append(obs)
        buffer.actions.append(action)
        buffer.log_probs.append(dist.log_prob(action))
        buffer.rewards.append(torch.tensor(reward, device=device))
        buffer.values.append(value)
        buffer.dones.append(torch.tensor(done, device=device))
        buffer.hiddens.append(h)

        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
        h = h_next.detach() 

        if done:
            break

    with torch.no_grad():
        last_value = value_net(h[-1]).squeeze(-1)

    buffer.values.append(last_value)
    return buffer

def ppo_update(
    policy_net,
    value_net,
    optimizer,
    buffer,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
):
    actions = torch.stack(buffer.actions)
    old_log_probs = torch.stack(buffer.log_probs)
    rewards = torch.stack(buffer.rewards)
    dones = torch.stack(buffer.dones)
    values = torch.stack(buffer.values)

    advantages, returns = compute_gae(rewards, values, dones)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = returns.unsqueeze(-1)
    
    policy_loss = 0.0
    value_loss = 0.0
    entropy_loss = 0.0

    T = len(actions)
    if DEBUG:
        print('return shape: ', returns.shape)


    
    for t in range(T):
        obs = buffer.obs[t].view(1, 1, -1)
        h = buffer.hiddens[t]

        probs, _ = policy_net(obs, h)
        probs = probs.squeeze(0).squeeze(0)

        dist = Categorical(probs)
        log_prob = dist.log_prob(actions[t])
        entropy = dist.entropy()

        ratio = torch.exp(log_prob - old_log_probs[t])

        surr1 = ratio * advantages[t]
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[t]

        policy_loss += -torch.min(surr1, surr2)
        entropy_loss += entropy

        value_pred = value_net(h[-1]).squeeze(-1)
        if DEBUG:
            print("returns: ", returns[t], " value pred: ", value_pred)
            time.sleep(SLEEP_DEBUG_TIME)
        value_loss += F.mse_loss(value_pred, returns[t])

    loss = (
        policy_loss / T
        + vf_coef * value_loss / T
        - ent_coef * entropy_loss / T
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(policy_net.parameters()) + list(value_net.parameters()), 0.5
    )
    optimizer.step()

def train(stock_tickers: list[str], num_episodes=500, save=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = RNNPolicy(
            input_dim=2,
            hidden_dim=64,
            num_actions=3,
            num_layers=2,
        ).to(device)

    value_net = DVN(hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(
            list(policy_net.parameters()) + list(value_net.parameters()),
            lr=3e-4,
        )

    for stock_ticker in stock_tickers:
        print(f'Training on {stock_ticker}')
        
        env = StockEnv(
            ticker=stock_ticker,
            start_date="2020-01-01",
            end_date="2025-01-01",
            render_mode=None,
        )

        for epoch in range(num_episodes):
            buffer = collect_rollout(
                env,
                policy_net,
                value_net,
                rollout_len=64,
                device=device,
            )

            ppo_update(
                policy_net,
                value_net,
                optimizer,
                buffer,
            )

            avg_reward = torch.stack(buffer.rewards).mean().item()
            print(f"Epoch {epoch:04d} | Avg reward: {avg_reward:.4f}")
        
    if save:
        torch.save(policy_net.state_dict(), RPOPATH)
        torch.save(value_net.state_dict(), VALUEPATH)

if __name__ == "__main__":
    
    stock_tickers = [
        'AAPL',
        'NVDA',
        'MSFT',
        'TSLA',
        'CETX',
        'BNAI',
        'COKE'
    ]
    
  
    train(stock_tickers=stock_tickers, num_episodes=1000, save=True)
