from gensim.models.keyedvectors import KeyedVectors
from huggingface_hub import hf_hub_download
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


# Params
matplotlib.rc('font', family='Noto Sans CJK JP')
words_path = 'words.tsv'
mode_repo_id = "Word2vec/wikipedia2vec_jawiki_20180420_300d"
model_filename = "jawiki_20180420_300d.txt"
n_components = 2
n_clusters = [3, 4, 5,]

# Load words
with open(words_path) as f:
    lines = f.readlines()
    # 단어	単語	単語（修正）	重要度	重要度の理由
    words = [line.split("\t") for line in lines[1:]]

# Get embedder
model = KeyedVectors.load_word2vec_format(hf_hub_download(repo_id=mode_repo_id, filename=model_filename, local_dir="/tmp/hf_hub"))

# Word check
def check_word(model: KeyedVectors, word: str):
    try:
        model[word]
    except Exception as e:
        print(f"Word not found: {word}")
        raise e

for i, word in enumerate(words):
    print(f'#{i+1}')
    check_word(model, word[2])

# Transform words to embeddings
embeddings = np.array([model[word[2]] for word in words])

# Reduce embeddings to 2D
reductor = TSNE(n_components=n_components)
reduced_embeddings = reductor.fit_transform(embeddings)

# Run clustering (e.g. Kmeans)
for n in n_clusters:
    clusterer = KMeans(n_clusters=n)
    clusterer.fit(reduced_embeddings)

    importance_font_size_map = {
        "1": 10,
        "2": 15,
        "3": 25,
    }

    # Visualize
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labeltop=False,
        labelbottom=False,) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the left edge are off
        right=False,         # ticks along the right edge are off
        labelleft=False,
        labelright=False,) # labels along the bottom edge are off
    plt.title(f'101 words({n} clusters)')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusterer.labels_)
    [plt.annotate(
        words[i][2],
        (reduced_embeddings[:, 0][i] + 0.1, reduced_embeddings[:, 1][i] + 0.1),
        fontsize=importance_font_size_map[words[i][3]])
        for i in range(len(reduced_embeddings))]
    plt.show()

print("Done")
