from agnt import RLAgent 
from llms import LLM 
from envr import Environment 


def main():
    # Initialize RL agent and environment

    #action_space = ["calculator", "calendar", "wikisearch", "translator"]
    action_space = ["add", "sub", "mul", "div"]
    
    rl_agent = RLAgent(action_space)
    llm = LLM() 
    environment = Environment()


    num_episodes = 10
    for episode in range(num_episodes):

        for state in environment: 

            q = state['question']
            q = q[2:-1].strip()

            a = state['answer']
            a = a[2:-1].strip() 

            perplexity_before = llm.calculate_perplexity(q)
            print(f"<main> Current state (ppt {round(perplexity_before, 4)}) = {q}")

            # Choose action (w/ optional params) based on agent's policy
            action, params = rl_agent.choose_action(q)
            print(f"<main> RL Agent chose: {action} w/ {params}")

            tool_name = None

            if action == 0:
                tool_name = 'add'
            elif action == 1:
                tool_name = 'sub'
            elif action == 2:
                tool_name = 'mul'
            elif action == 3:
                tool_name = 'div'
            else:
                raise ValueError("Invalid action value. Expected values: 0, 1, 2, 3")

            params1, params2 = params

            if params1 is not None and params2 is not None: 
                p1 = state[params1]
                p2 = state[params2]

                params = (p1, p2)

            # Perform action in the environment
            result = environment.perform_action(q, tool_name, params)


            # Calculate perplexity after using the tool
            perplexity_after = llm.calculate_perplexity(result)
            print(f"<main> Perplexity after {perplexity_after}")


            # Get reward
            reward = environment.get_reward(perplexity_before, perplexity_after)

            # Update RL agent's policy based on observed experience
            rl_agent.update_policy(q, action, params, result, reward)


if __name__ == '__main__': 
    main()
