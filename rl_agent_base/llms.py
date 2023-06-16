import torch
from cnst import DBG
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class LLM(): 
    def __init__(self): 

        # Load pre-trained GPT-2 model and tokenizer
        model_name = 'gpt2'
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Set up device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if DBG: print(f"<llms> Set device to: {device}")

        self.model.to(device)
        self.model.eval()


    # Calculate perplexity for a given sentence
    def calculate_perplexity(self, sentence):

        # Tokenize input sentence
        input_ids = self.tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt").to(device)

        with torch.no_grad():
            # Generate predicted tokens
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits

        # Calculate perplexity
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        perplexity = torch.exp(loss)

        p_p_t = perplexity.item() / input_ids.numel()
        if DBG: print(f"<llms> Perplexity-Per-Token = {p_p_t} ({sentence})")
