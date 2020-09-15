import networkx as nx
from node2vec import Node2Vec

# FILES
EMBEDDING_FILENAME = './embeddings/embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings/embeddings.model'
EDGELIST_INPUT = './edgelist/Biogrid.edgelist'

# Create a graph
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)

graph = nx.read_edgelist(EDGELIST_INPUT, nodetype=int, create_using=nx.DiGraph())
# for edge in graph.edges():
#     graph[edge[0]][edge[1]]['weight'] = 1

# Precompute probabilities and generate walks
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=1)

## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
#node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

# Embed
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)
