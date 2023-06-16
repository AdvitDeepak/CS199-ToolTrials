import torch 
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class ToolPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ToolPredictionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out
    

class ParameterDecisionModel(nn.Module):
    def __init__(self):
        super(ParameterDecisionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 1)

    def forward(self, input_sentence):
        input_tensor = self.tokenizer.encode_plus(input_sentence, add_special_tokens=True, padding='longest', return_tensors='pt')['input_ids']
        outputs = self.bert(input_tensor)
        last_hidden_state = outputs.last_hidden_state[:, 1:-1, :]  # Exclude [CLS] and [SEP] tokens
        logits = self.linear(last_hidden_state)
        return logits


class RLAgent:
    def __init__(self, tools, input_size=768, hidden_size=26, param_size=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tool_size = len(tools)
        self.param_size = param_size
        
        # Define the neural network models for tool selection and parameter selection
        self.tool_model = ToolPredictionModel(input_size=768, hidden_size=256, num_layers=2, output_size=4) 
        self.param_model = ParameterDecisionModel()

        # Define the optimizers
        self.tool_optimizer = optim.Adam(self.tool_model.parameters(), lr=0.001)
        self.param_optimizer = optim.Adam(self.param_model.parameters(), lr=0.001)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')



    def choose_action(self, state):
        # Convert the state to a PyTorch tensor

        tokens = self.tokenizer.encode(state, add_special_tokens=True)
        input_tensor = torch.tensor(tokens)
        input_tensor = input_tensor.unsqueeze(0)
        outputs = self.model(input_tensor)

        output_tensor = outputs.last_hidden_state

        print(output_tensor)
        print(output_tensor.shape)

        # Forward pass through the tool selection model
        tool_probs = self.tool_model(output_tensor)
        print(tool_probs.shape)
        tool_probs = tool_probs.view(-1)

        # Choose tool based on tool probabilities
        tool = torch.multinomial(tool_probs, num_samples=1).item()
        
        # Forward pass through the parameter selection model
        param_probs = self.param_model(state)
        param_indices = torch.topk(param_probs.squeeze(), k=2).indices

        print(param_probs)
        print(param_probs.shape)
        print(param_indices)

        tokens = self.tokenizer.convert_ids_to_tokens(param_indices)
        characters = []
        for token in tokens:
            if token not in self.tokenizer.all_special_tokens and token in state:
                characters.append(token)

        print(characters)
        
        print("Chose tool", tool)

        if len(characters) != 0: 
            param1 = characters[0]
            param2 = characters[1]
        else: 
            param1 = None 
            param2 = None 

        return tool, (param1, param2)
    

    def update_policy(self, state, tool, param1, param2, reward):
        # Convert the state to a PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # Convert the tool, param1, and param2 to PyTorch tensors
        tool_tensor = torch.tensor(tool, dtype=torch.long)
        param1_tensor = torch.tensor(param1, dtype=torch.long)
        param2_tensor = torch.tensor(param2, dtype=torch.long)
        
        # Convert the reward to a PyTorch tensor
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        
        # Forward pass through the tool selection model
        tool_probs = self.tool_model(state_tensor)
        
        # Calculate the log probability of the chosen tool
        tool_log_prob = torch.log(tool_probs.squeeze(0)[tool_tensor])
        
        # Forward pass through the parameter selection model
        param_probs = self.param_model(state_tensor)
        
        # Calculate the log probability of the chosen parameters
        param1_log_prob = torch.log(param_probs.squeeze(0)[param1_tensor])
        param2_log_prob = torch.log(param_probs.squeeze(0)[param2_tensor])
        
        # Calculate the loss as the negative sum of log probabilities multiplied by the reward
        loss = -(tool_log_prob + param1_log_prob + param2_log_prob) * reward_tensor
        
        # Backpropagation and optimization step for tool selection
        self.tool_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.tool_optimizer.step()
        
        # Backpropagation and optimization step for parameter selection
        self.param_optimizer.zero_grad()
        loss.backward()
