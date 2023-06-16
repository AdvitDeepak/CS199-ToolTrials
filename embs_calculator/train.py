import torch
import torch.nn as nn
from transformers import BertModel


class BERTTrainer:
    def __init__(self, encoded_pairs):
        self.encoded_pairs = encoded_pairs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def build_model(self, tokenizer):
        # Build the BERT model architecture
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.model.resize_token_embeddings(len(tokenizer))

    def prepare_training_data(self):
        # Prepare the encoded input-output pairs for training
        input_ids = []
        attention_masks = []
        token_type_ids = []

        for pair in self.encoded_pairs:
            input_el, output_el = pair 
            input_id = torch.tensor(input_el).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor([1] * len(pair)).unsqueeze(0).to(self.device)
            token_type_id = torch.tensor([0] * len(pair)).unsqueeze(0).to(self.device)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)

        return input_ids, attention_masks, token_type_ids

    def train_model(self, input_ids, attention_masks, token_type_ids, num_epochs=10):
        # Train the BERT model using the prepared training data
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        criterion = nn.MSELoss()

        labels = {
            "x0x" : 30522, 
            "x1x" : 30523, 
            "x2x" : 30524, 
            "x3x" : 30525,
        }

        for epoch in range(num_epochs):
            total_loss = 0

            for i in range(len(input_ids)):
                optimizer.zero_grad()

                input_id = input_ids[i]

                indices = torch.nonzero(input_id > 30521)
                #print(input_id)
               # print(indices)
                # Get the index of the first occurrence
                first_index = indices[0][0]

                attention_mask = attention_masks[i]
                token_type_id = token_type_ids[i]
                #print(input_id)
                outputs = self.model(input_id)
                predicted_output = outputs.last_hidden_state.squeeze(0).mean(dim=0)

                target_output = torch.tensor(self.encoded_pairs[i][1], dtype=torch.float).to(self.device)
                
                #print("Outputs", type(outputs))
                #print("Outputs", len(outputs))
                #print(len(outputs.last_hidden_state.squeeze(0)))
                #print(outputs[0][:, first_index, :])
                #print("Predicted_outputs", predicted_output.shape)
                #print("   Target outputs", target_output.shape)


                target_output = torch.squeeze(outputs[0][:, first_index, :]).to(self.device)

                loss = criterion(predicted_output, target_output)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch: {epoch + 1} | Loss: {total_loss / len(input_ids):.4f}")

    def save_embeddings(self, save_path):
        # Extract and save the learned embeddings for each mystery function
        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for pair in self.encoded_pairs:
                encoded_input, _ = pair
                input_id = torch.tensor(encoded_input).unsqueeze(0).to(self.device)

                outputs = self.model(input_id)
                embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy().tolist()
                embeddings.append(embedding)

        torch.save(embeddings, save_path)
