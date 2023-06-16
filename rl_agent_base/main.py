from agnt import RlAgent 
from llms import LLM 
from envr import Environment 
from cnst import DBG 


def main():
    # Initialize RL agent and environment

    #action_space = ["calculator", "calendar", "wikisearch", "translator"]
    action_space = ["add", "sub", "mul", "div"]
    
    rl_agent = RlAgent(action_space)
    llm = LLM() 
    environment = Environment()


    num_episodes = 1000
    for episode in range(num_episodes):

        for state in environment: 
            perplexity_before = llm.calculate_perplexity(result)
            print(f"<main> Current state (ppt {perplexity_before}) = {state}")


            # Choose action (w/ optional params) based on agent's policy
            action, params = rl_agent.choose_action(state)
            print(f"<main> RL Agent chose: {action} w/ {params}")


            # Perform action in the environment
            result = environment.perform_action(state, action, params)


            # Calculate perplexity after using the tool
            perplexity_after = llm.calculate_perplexity(result)
            print(f"<main> Perplexity after {perplexity_after}")


            # Get reward
            reward = environment.get_reward(perplexity_before, perplexity_after)

            # Update RL agent's policy based on observed experience
            rl_agent.update_policy(state, action, params, result, reward)


if __name__ == '__main__': 
    main()
