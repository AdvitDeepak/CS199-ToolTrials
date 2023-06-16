import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the embeddings from the embeddings.pth file
embeddings_list = torch.load("embeddings.pth")
embeddings = [torch.tensor(embedding).reshape(1, -1) for embedding in embeddings_list]

# # Compute cosine similarity across all combinations of input pairs
similarity_matrix = torch.zeros((4, 4))

pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

labels = ["add", "sub", "mul", "div"]

for pair in pairs: 
    i, j = pair
    emb1 = embeddings[i]
    emb2 = embeddings[j]
    similarity = torch.cosine_similarity(emb1, emb2)
    similarity_matrix[i][j] = similarity

    print(f"Func{i} ({labels[i]}) & Func{j} ({labels[j]}) ==> {similarity}")

# # Visualize the embeddings in 3D space
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for i in range(4):
#     emb = embeddings_list[i]
#     ax.scatter(emb[0], emb[1], emb[2], label=f"FUNC_{i+1}")

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()

# plt.show()

# # Print the cosine similarity matrix
# print("Cosine Similarity Matrix:")
# print(similarity_matrix)

import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords

# Load the pretrained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# # Load the learned embeddings for mystery functions
# embeddings = torch.load("embeddings.pth")

# # Convert embeddings to tensors
# embeddings = [torch.tensor(embedding) for embedding in embeddings]

# Get the list of vocabulary words in BERT
vocab_words = list(tokenizer.vocab.keys())
stopwords = set(stopwords.words('english'))

top_similar_words = [] 

# Find the top 10 words in BERT's vocabulary closest to each mystery function embedding
for i, embedding in enumerate(embeddings):
    # Compute cosine similarity between the embedding and all vocabulary word embeddings

    #print(type(embedding))
    #print(type(embedding.detach().numpy()))
    #print(type(model.get_input_embeddings().weight))

    similarity_scores = cosine_similarity(embedding.detach().numpy(), model.get_input_embeddings().weight.detach().numpy())

    # Get the indices of the top 10 most similar words
    top_indices = similarity_scores.argsort()[0][-300:]

    # Get the actual words from the indices
    top_words = [vocab_words[idx] for idx in top_indices if vocab_words[idx] not in stopwords]


    # Print the mystery function and the top 10 closest words
    #print(f"Mystery Function {i+1}:")

    top_similar_words.append(top_words)

    #print()

common_elements = set(top_similar_words[0]).intersection(*top_similar_words[1:])

# Remove the common elements from each array
filtered_arrays = [[elem for elem in arr if elem not in common_elements] for arr in top_similar_words]

for entry in filtered_arrays: 
    print(entry)
