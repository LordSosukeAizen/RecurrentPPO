import torch
from rppo import RNNPolicy, DVN
from stockenv import StockEnv

def load_model(device):
    policy_net = RNNPolicy(
        input_dim=2,
        hidden_dim=64,
        num_actions=3,
        num_layers=2,
    ).to(device)

    value_net = DVN(hidden_dim=64).to(device)

    policy_checkpoint = torch.load("recurrent_po.pth")
    policy_net.load_state_dict(policy_checkpoint)
    
    value_checkpoint = torch.load("value.pth")
    value_net.load_state_dict(value_checkpoint)

    policy_net.eval()
    value_net.eval()

    return policy_net, value_net

@torch.no_grad()
def run_episode(env, policy_net, device, render=False):
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    h = torch.zeros(
        policy_net.num_layers, 1, policy_net.hidden_dim, device=device
    )

    done = False
    total_reward = 0.0
    step = 0

    while not done:
        obs_seq = obs.view(1, 1, -1)  # (T=1, B=1, obs_dim)

        probs, h_next = policy_net(obs_seq, h)
        probs = probs.squeeze(0).squeeze(0)

        action = torch.argmax(probs).item()

        obs, reward, done, _, _ = env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

        total_reward += reward
        h = h_next   # no detach needed (no grad)

        if render:
            env.render()

        step += 1

    return total_reward, step

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = StockEnv(
        ticker="GOOGL",
        start_date="2025-03-01",
        end_date="2025-06-01",
        render_mode="human",  # set None for headless
    )

    policy_net, _ = load_model(
        device=device
    )

    total_reward, steps = run_episode(
        env,
        policy_net,
        device,
        render=True,
    )

    print(f"Test episode finished")
    print(f"Steps taken: {steps}")
    print(f"Total reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test()
