from old_rl_agent.tool import Calculator, Calendar, WikiSearch, Translator 
from help import read_txt_file
from cnst import DATA_PATH
from datasets import load_dataset


"""

TODO: 
- train.txt contains dummy data --> change data loading process to actuallly
  use a big boi corpus w/ unstructured, not necessarily math text (GPT-2 trainset)

"""

# Define environment
class Environment:
    def __init__(self):
        self.calculator = Calculator()
        #self.calendar = Calendar()
        #self.wikisearch = WikiSearch()
        #self.translator = Translator()

        #self.data, self.num = read_txt_file(DATA_PATH)
        #self.cntr = 0
        #self.cols = 0

        self.dataset = load_dataset('math_dataset', 'arithmetic__mixed', split='train')
        self.num = len(self.dataset)


    # Return the current state based on the environment's context
    # THIS STATE SHOULD BE THE INPUT SENTENCE 
    def __getitem__(self, idx):

        # val = self.data[self.cntr][0:self.cols]

        # if self.cols > len(self.data[self.cntr]): 
        #     self.cntr += 1
        #     self.cols = 0 

        # return val 

        return self.dataset[idx]
    
    

    # Helper function to ensure __getitem__ works properly 
    def __len__(self): 
        return self.num 
    


    def perform_action(self, state, action, params):
        context = None 
        param1, param2 = params 

        if action == "add":
            context = self.calculator.add(param1, param2)
        elif action == "sub":
            context = self.calculator.sub(param1, param2)
        elif action == "mul":
            context = self.calculator.mul(param1, param2)
        elif action == "div":
            context = self.calculator.div(param1, param2)


        # if action == "calculator":
        #     context = self.calculator.calculate(params)
        # elif action == "calendar":
        #     context = self.calendar.schedule(params)
        # elif action == "wikisearch":
        #     context = self.wikisearch.search(params)
        # elif action == "translator":
        #     context = self.translator.translate(params)

        if context: 
            return f"Given that {context}, {state}" # IF WE ARE USING GPT, and NOT DOLLY2
        else: 
            return state 


    def get_reward(self, perplexity_before, perplexity_after):
        return perplexity_before - perplexity_after