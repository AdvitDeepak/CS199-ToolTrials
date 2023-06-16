import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class LLM(): 
    def __init__(self): 

        # Load pre-trained GPT-2 model and tokenizer
        model_name = 'gpt2'
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Set up device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"<llms> Set device to: {self.device}")

        self.model.to(self.device)
        self.model.eval()


    def tokenize(self, sentence):
        return self.tokenizer.encode(sentence, return_tensors="pt").to(self.device) 

    # Calculate perplexity for a given sentence
    def calculate_perplexity(self, sentence):

        # Tokenize input sentence
        input_ids = self.tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Generate predicted tokens
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits

        # Calculate perplexity
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        perplexity = torch.exp(loss)

        p_p_t = perplexity.item() / input_ids.numel()
        print(f"<llms> Perplexity-Per-Token = {p_p_t} ({sentence})")

        return p_p_t
