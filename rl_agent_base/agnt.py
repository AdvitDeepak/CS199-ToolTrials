from cnst import DBG
import torch 

# Define RL Agent
class RlAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Define the neural network model
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
        
        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def choose_action(self, state):
        if DBG: print(f"Current state: {state}")
        
        # Convert the state to a PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # Forward pass through the model
        action_probs = self.model(state_tensor)
        
        # Choose action based on action probabilities
        action = torch.multinomial(action_probs, num_samples=1).item()
        
        return action


    def update_policy(self, state, action, reward, next_state):
        # Convert the state to a PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # Convert the action to a PyTorch tensor
        action_tensor = torch.tensor(action, dtype=torch.long)
        
        # Convert the reward to a PyTorch tensor
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        
        # Forward pass through the model
        action_probs = self.model(state_tensor)
        
        # Calculate the log probability of the chosen action
        log_prob = torch.log(action_probs.squeeze(0)[action_tensor])
        
        # Calculate the loss as the negative log probability multiplied by the reward
        loss = -log_prob * reward_tensor
        
        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
