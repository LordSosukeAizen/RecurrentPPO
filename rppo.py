import torch
import torch.nn as nn
import torch.nn.functional as F



class PolicyNetwork(nn.Module):
    '''
    Policy Network generates probability distribution by taking an hidden state from underlying RNN
    '''
    def __init__(self, hidden_dim, num_actions):
        
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_actions)

    def forward(self, h):
        x = F.relu(self.fc1(h))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)


class RNNPolicy(nn.Module):
    '''
    RNNPolicy represents the hidden state by taking stock price and stocks hold which are observation
    and combining with previous hidden state to represent hidden state which eventually is given 
    to a policy network.
    '''
    def __init__(self, input_dim, hidden_dim, num_actions, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
        )

        self.policy = PolicyNetwork(hidden_dim, num_actions)

    def forward(self, x, h0=None):
        """
        x: (T, B, input_dim)
        h0: (num_layers, B, hidden_dim)
        """
        T, B, _ = x.shape

        if h0 is None:
            h0 = torch.zeros(self.num_layers, B, self.hidden_dim)

        # RNN forward
        h_seq, hT = self.rnn(x, h0)  # (T, B, H)

        # Apply policy at each timestep
        probs = []
        for t in range(T):
            p = self.policy(h_seq[t])  # (B, num_actions)
            probs.append(p)

        probs = torch.stack(probs)  # (T, B, num_actions)
        return probs, hT
    
    
class DVN(nn.Module):
    '''
    Deep Value Network
    '''
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, h):
        """
        h: (..., hidden_dim)
        returns: (..., 1)
        """
        x = F.relu(self.fc1(h))
        return self.fc2(x)