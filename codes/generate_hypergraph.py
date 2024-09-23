import pandas as pd
import numpy as np 
import networkx as nx
import pickle as pkl

dt = 'risk-models-benchmark-v1'

df = pd.read_csv('./datasets/' + dt + '.csv')

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentences_statement1 = df['Statement 1']
sentences_statement2 = df['Statement 2']

embeddings_1 = model.encode(sentences_statement1)
embeddings_2 = model.encode(sentences_statement2)

df['Embedding_Stt_1'] = list(embeddings_1)
df['Embedding_Stt_2'] = list(embeddings_2)

graph = nx.DiGraph()

for _, row in df.iterrows(): 
    graph.add_edge(row['Statement 1'], 'Node_edge:' + row['Statement 1']  + ' + ' + row['Statement 2']) 
    graph.add_edge('Node_edge:' + row['Statement 1']  + ' + ' + row['Statement 2'],row['Statement 2'])

    emb_1 = np.asarray(row['Embedding_Stt_1'], dtype=np.float64)
    emb_2 = np.asarray(row['Embedding_Stt_2'], dtype=np.float64)
    
    graph.nodes[row['Statement 1']]['embedding'] = emb_1
    graph.nodes[row['Statement 2']]['embedding'] = emb_2

    graph.nodes['Node_edge:' + row['Statement 1']  + ' + ' + row['Statement 2']]['embedding'] = np.mean([emb_1,emb_2], axis=0)

    graph.nodes[row['Statement 1']]['label'] = 'aux'
    graph.nodes[row['Statement 2']]['label'] = 'aux'

    graph.nodes['Node_edge:' + row['Statement 1']  + ' + ' + row['Statement 2']]['label'] = row['label']


with open('./datasets/' + dt + '_digraph.pkl', "wb") as file:
     pkl.dump(graph, file)