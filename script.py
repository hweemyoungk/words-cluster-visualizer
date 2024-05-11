from gensim.models.keyedvectors import KeyedVectors
from huggingface_hub import hf_hub_download


# Load words

# Get embedder
model = KeyedVectors.load_word2vec_format(hf_hub_download(repo_id="Word2vec/wikipedia2vec_jawiki_20180420_300d", filename="jawiki_20180420_300d.txt"))

# Transform words to embeddings
model['使命']

# Run clustering (e.g. Kmeans)

# Reduce embeddings to 2D

# Visualize