# COM SCI 199 - Language Models and Tools

> Advit Deepak, Spring '23, University of California, Los Angeles (UCLA)

Codebase for experiments conducted in my quarter-long research project as part of COM SCI 199. 

*This report explores the use of tools, or functions that accomplish a certain task, with Large Language Models (LLMs). The inspiration behind this direction of research is the paper Toolformer: Language Models Can Teach Themselves to Use Tools, which presents an in-context learning approach to enable functionalities such as GPT-4's usage of plugins. This report examines the methodology utilized and experiments with two novel approaches to incorporate plugin functionality with smaller language models. With a tool embedding approach, I attempt to enable Language Models to learn representations of functions as part of their vocabulary. With a Reinforcement Learning (RL) agent approach, I attempt to train a smaller model which can query the appropriate tools with the appropriate inputs based on contextual cues. By avoiding an in-context learning approach, the outcomes of this research provide insights into the feasibility and effectiveness of extending the plugin functionality to smaller language models.*

This repository is an archive of ongoing experiments. As a result, it contains several works-in-progres. 

> A detailed report regarding these experiments can be found at [<hyperlink coming soon\>](). 

&nbsp; &nbsp;

* * *

&nbsp; &nbsp;

## Hirearchy of Repository 

- assets - contains all images utilized in the final report, hyperlinked above 
- embs_calculator - experiments regarding learning static embedings for add, sub, mul, div
- embs_thesaurus - experiments regarding learning contextual embeddings for synonym, antonym
- rl_agent_base - base agent, environment, action, and reward defintion for Toolformer replication
- rl_agent_exps - offshoot of base agent for add, sub, mul, div using perplexity-based reward 
- toolformer - base toolformer-pytorch from [lucidrain's implementation](https://github.com/lucidrains/toolformer-pytorch)
> Detailed descriptions regarding these folders can be found in the Appendix of [<hyperlink coming soon\>](). 

&nbsp; &nbsp;

## Creating Environment

To create and activate the conda environment required to run these experiments, kindly execute: 

``` 
conda env create -f 199_env.yml 
```


