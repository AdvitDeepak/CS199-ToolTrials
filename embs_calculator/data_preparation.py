import numpy as np
import random 

from tools import MysteryTools 
from transformers import BertTokenizer


class DataProcessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        special_words = ["X0X", "X1X", "X2X", "X3X"]
        self.tokenizer.add_tokens(special_words)
    
    def generate_input_output_pairs(self, num_pairs, input_vector):
        
        pairs = []

        for _ in range(num_pairs):
            a = random.randint(-1000, 1000)
            b = random.randint(-1000, 1000)
            pairs.append((a, b))


        data = []
        mystery_tool = MysteryTools()

        for i in range(num_pairs):
            input_a, input_b = pairs[i]
            for j in range(4): 

                # Generate data for FUNC_1 (add)
                output = mystery_tool.get_func(j)(input_a, input_b)
                res = f"{input_a} {input_b} X{j}X"
                data.append((res, output))

        return data
    

    
    def tokenize_input_output_pairs(self, pairs):
        # Tokenize the input-output pairs
        tokenized_pairs = []

        for pair in pairs:
            input_val, output_val = pair
            tokenized_input = self.tokenizer.tokenize(str(input_val))
            tokenized_output = self.tokenizer.tokenize(str(output_val))
            tokenized_pair = (tokenized_input, tokenized_output)
            tokenized_pairs.append(tokenized_pair)

        return tokenized_pairs

    def encode_input_output_pairs(self, pairs):
        encoded_pairs = []

        for input_vals, output_vals in pairs:
            input_encoded = self.tokenizer.encode(input_vals, add_special_tokens=True, truncation=True)
            target_encoded = self.tokenizer.encode(output_vals, add_special_tokens=True, truncation=True)

            # Pad or truncate the target tensor to match the size of the input tensor
            if len(target_encoded) < len(input_encoded):
                target_encoded = target_encoded + [self.tokenizer.pad_token_id] * (len(input_encoded) - len(target_encoded))
            elif len(target_encoded) > len(input_encoded):
                target_encoded = target_encoded[:len(input_encoded)]

            encoded_pairs.append((input_encoded, target_encoded))

        return encoded_pairs, self.tokenizer