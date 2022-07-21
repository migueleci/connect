# %%

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import igraph as ig
from matplotlib import pyplot as plt

# %%

# --- Create network edgelist from ATTED data

path = 'raw data/zma-r-gcn'
files = os.listdir(path)

edgelist = list()
geneIdx = dict((x,i) for i,x in enumerate(files))
threshold = 2

# %%

nV, nE = len(geneIdx.keys()), 0

file = open('data/gcn_edgelist.csv', 'w')
for gene in tqdm(files):
  data = pd.read_csv('{0}/{1}'.format(path,gene), delimiter='\t', names = ['gene', 'score'])
  data = data.astype({'gene': str, 'score': np.float64})
  data = data[(data['score'] > threshold)]

  for row in data.to_dict('records'):
    v, s = row['gene'], row['score']
    if geneIdx[gene] < geneIdx[v]:
      nE += 1
      file.write('{0},{1},{2}\n'.format(geneIdx[gene], geneIdx[v], s))
file.close()

file = open('data/genes.txt','w')
file.write('\n'.join(list(geneIdx.keys())))
file.close()

print("Number of genes in GCN: {0}".format(nV))
print("Gene co-expression relations: {0}".format(nE))

# %%

# --- Modify edge weight (zcore)

file = open('data/genes.txt','r')
genes = [x.strip() for x in file.readlines()]
file.close()

data = pd.read_csv('data/gcn_edgelist.csv', names=['source','target','score'])
data = data.astype({'source': int, 'target': int, 'score': np.float64})

smax = data['score'].max()
data['score'] = (smax - data['score']) + 1
data.to_csv('data/edgelist.csv', index=False)

data['score'].hist(bins=10)
plt.savefig('zcore.pdf', format='pdf', dpi=600)

print('{0:.2f}% of the total number of edges are in the GCN'.format((nE*100)/(nV*nV)))

# %%

# ---- Get giant connected component

g = ig.Graph()
g.add_vertices(nV)
g.vs["name"] = genes

edges = list(data[['source','target']].itertuples(index=False, name=None))
g.add_edges(edges)
print(ig.summary(g))
print(g.is_connected())

# %%

file = open("raw data/ENTREZ_GENE_ID2GOTERM_BP_ALL.txt", "r")
gene_term = list()
for line in file.readlines():
  g, t = line.strip().split('\t')
  t, d = t.split("~")
  gene_term.append((g,t))
gene_term_new = pd.DataFrame(gene_term, columns=["Gene","Term"])
len(gene_term_new), len(gene_term_new.Term.unique())
gene_term_new.to_csv("data/gene_term_raw.csv", index=False)
print(gene_term_new)
