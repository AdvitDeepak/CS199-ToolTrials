import torch 
import csv 

from data_preparation import DataProcessor
from model_training import BERTTrainer
from transformers import BertTokenizer


def main():
    # Data preparation
    data_processor = DataProcessor()
    input_vector = [1, 2, 3, 4, 5]  # Example input vector
    num_pairs_per_function = 1  # Example number of pairs per function
    


    print("\n\n=====================================================\n\n")
    pairs = data_processor.generate_input_output_pairs(num_pairs_per_function, input_vector)
    print(pairs)    
    tokenized_pairs = data_processor.tokenize_input_output_pairs(pairs)
    print(tokenized_pairs)
    encoded_pairs, tokenizer = data_processor.encode_input_output_pairs(tokenized_pairs)
    print(encoded_pairs)
    print("\n\n=====================================================\n\n")

    # filename = "data.csv"
    # vals = []

    # with open(filename, "r") as i:
    #     reader = csv.reader(i, delimiter=",")
    #     for _, abstract in reader:
    #         vals.append(abstract)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenized_entries = []

    # for entry in vals:
    #     tokens = tokenizer.tokenize(entry)
    #     tokens = tokenizer.encode(tokens, add_special_tokens=True, truncation=True)
    #     tokenized_entries.append(tokens)

    # Model training
    trainer = BERTTrainer(encoded_pairs)
    trainer.build_model(tokenizer)
    input_ids, attention_masks, token_type_ids = trainer.prepare_training_data()
    trainer.train_model(input_ids, attention_masks, token_type_ids, num_epochs=15)

    # Save embeddings
    save_path = 'embeddings.pth'  # Example save path
    trainer.save_embeddings(save_path)




if __name__ == '__main__':
    torch.cuda.empty_cache()  # Clear GPU cache
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # Set the device to the first available GPU (change the index if needed)
        with torch.cuda.device(device):
            main()
    else:
        main()