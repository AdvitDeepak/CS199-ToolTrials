from tool import Calculator
from datasets import load_dataset


# Define environment
class Environment:
    def __init__(self):
        self.calculator = Calculator()
        self.dataset = load_dataset('math_dataset', 'arithmetic__mixed', split='train')
        self.num = len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]
    

    # Helper function to ensure __getitem__ works properly 
    def __len__(self): 
        return self.num 
    


    def perform_action(self, state, action, params):
        context = None 
        param1, param2 = params 
        print("<envr> Performing action", action)
        bad_params = param1 is None or param2 is None

        if bad_params: 
            param1 = 1
            param2 = 1

        if action == "add":
            context = self.calculator.add(param1, param2)
        elif action == "sub":
            context = self.calculator.sub(param1, param2)
        elif action == "mul":
            context = self.calculator.mul(param1, param2)
        elif action == "div":
            context = self.calculator.div(param1, param2)

        if context: 
            if bad_params: 
                print("retuning no params!")
                return f"Given that we {action}, {state}"
            
            return f"Given that {param1} {action} {param2} is {context}, {state}" # IF WE ARE USING GPT, and NOT DOLLY2
        else: 
            return state 


    def get_reward(self, perplexity_before, perplexity_after):
        return perplexity_before - perplexity_after