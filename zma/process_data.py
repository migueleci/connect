import os
import sys
import multiprocessing

import numpy as np
import igraph as ig
import pandas as pd
import networkx as nx

from tqdm import tqdm
from time import time
from collections import deque
from matplotlib import pyplot as plt

from HBN import *

from goatools.obo_parser import GODag
from goatools.semantic import deepest_common_ancestor, common_parent_go_ids
from goatools.godag.go_tasks import get_go2parents
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.gosubdag.plot.gosubdag_plot import GoSubDagPlot

# Node embedding, cross-validation and scaler
from pecanpy import pecanpy, node2vec
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

# Ploting
from matplotlib import rc

rc('font', family='serif', size=18)
rc('text', usetex=False)

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# %%

g = GODag("/home/miguel/projects/omics/transfer_data/zma_data/data/go-basic.obo")

file = open('/home/miguel/projects/omics/transfer_data/zma_data/data/genes.txt','r')
genes = [x.strip() for x in file.readlines()]
file.close()

g2t = pd.read_csv("/home/miguel/projects/omics/transfer_data/zma_data/data/gene_term_raw.csv", dtype=object)
g2t = g2t[g2t.Gene.isin(genes)].reset_index(drop=True)

terms = g2t.Term.unique().tolist()
terms = np.array([t for t in terms if t in g and g[t].namespace == "biological_process" and t != "GO:0008150"])
nA = len(terms)

g2t = g2t[g2t.Term.isin(terms)].reset_index(drop=True)
g2t.to_csv("/home/miguel/projects/omics/transfer_data/zma_data/data/gene_term.csv", index=False)

# %%

isa = list()
for t in tqdm(terms):
  q = deque()
  for p in g[t].parents:
    q.append((t, p.id))

  while len(q) > 0:
    c, p = q.pop()
    if p != "GO:0008150" and g[p].namespace == "biological_process":
      isa.append((c,p))
      for gp in g[p].parents:
        q.append((p, gp.id))

isa = pd.DataFrame(isa, columns=['Child','Parent'])
isa = isa.drop_duplicates().reset_index(drop=True)
# isa.to_csv("/home/miguel/projects/omics/transfer_data/zma_data/data/isa.csv", index=False)

all_terms = np.union1d(np.union1d(isa.Child, isa.Parent), terms)
term_def = pd.DataFrame()
term_def["Term"] = all_terms
term_def["Desc"] = [g[t].name for t in all_terms]
# term_def.to_csv("/home/miguel/projects/omics/transfer_data/zma_data/data/term_def.csv", index=False)

data_gcn = pd.read_csv("/home/miguel/projects/omics/transfer_data/zma_data/data/edgelist.csv")
nV, idV = len(genes), dict([(v,i) for i,v in enumerate(genes)])

terms = all_terms.copy()
nA, idA = len(terms), dict([(a,i) for i,a in enumerate(terms)])

# %%

# GCN matrix
# ng:number of genes, idg:gene index map
gcn = np.zeros((nV,nV))
for edge in tqdm(data_gcn.to_dict("records")):
  u, v, s = edge.values()
  gcn[u][v] = gcn[v][u] = s

# go by go matrix
# nt:number of terms, idt:term index map
go_by_go = np.zeros((nA,nA))
for edge in tqdm(isa.to_dict("records")):
  u, v = idA[edge["Child"]], idA[edge["Parent"]]
  go_by_go[u,v] = 1

# compute the transitive closure of the ancestor of a term (idx)
def ancestors(term):
  tmp = np.nonzero(go_by_go[term,:])[0]
  ancs = list()
  while len(tmp) > 0:
    tmp1 = list()
    for i in tmp:
      ancs.append(i)
      tmp1 += np.nonzero(go_by_go[i,:])[0].tolist()
    tmp = list(set(tmp1))
  return ancs

# gene by go matrix
gene_by_go = np.zeros((nV,nA))
for edge in tqdm(g2t.to_dict("records")):
  u, v = idV[edge["Gene"]], idA[edge["Term"]]
  gene_by_go[u,v] = 1
  gene_by_go[u,ancestors(v)] = 1

print()
print('**Maize data**')
print('Genes: \t\t{0:7}'.format(nV))
print('Co-expression: \t{0:7.0f}'.format(np.count_nonzero(gcn)/2))
print('GO terms: \t{0:7}'.format(nA))
print('GO hier.: \t{0:7.0f}'.format(np.sum(go_by_go)))
print('Gene annot.: \t{0:7}'.format(np.count_nonzero(gene_by_go)))

# %%

"""
                 **Osa**     **Zma**
Genes: 		        22698       18957 (<)
Co-expression: 	8771819     6548437 (<)
GO terms: 	       4668        5694 (>)
GO hier.: 	       8406       10366 (>)
Gene annot.: 	   344943      521738 (>)
"""

# %%

#####################################
# 2. Prepare term data for prediction
#####################################

# Graph for subhiearchies creation
g2g_edg = np.transpose(np.nonzero(np.transpose(go_by_go))).tolist()
g2g = nx.DiGraph()
g2g.add_nodes_from(np.arange(nA))
g2g.add_edges_from(g2g_edg)
print('GO graph (all terms): nodes {0}, edges {1}'.format(g2g.number_of_nodes(), g2g.number_of_edges()))
print('Number of weakly conn. components: {}'.format(nx.number_weakly_connected_components(g2g)))

# Prune terms according to paper, very specific and extremes with little to
# no information terms are avoided. Select genes used for prediction
# Accoding to restriction 5 <= genes annotated <= 300
ft_idx = list() # list of terms filtered according to the previous explanation
for i in range(nA):
  if 0 <= np.count_nonzero(gene_by_go[:,i]):
    ft_idx.append(i)
print('Number of filtered terms: {0}'.format(len(ft_idx)))

# Including the ancestor of the selected terms
pt_idx = list(ft_idx)
for i in ft_idx:
  pt_idx += np.nonzero(go_by_go[i,:])[0].tolist()
pt_idx = np.array(sorted(list(set(pt_idx))))
print('Number of filtered terms incl. parents: {0}'.format(len(pt_idx)))

file = open('/home/miguel/projects/omics/transfer_data/zma_data/data/terms.txt', 'w')
file.write('\n'.join(terms[pt_idx]))
file.close()
if int(sys.argv[1]):
  sys.exit()
# %%

file = open('/home/miguel/projects/omics/transfer_data/terms_inter.txt', 'r')
inter_terms = np.array([x.strip() for x in file.readlines()])
file.close()

pt_idx = np.array([np.where(terms == x)[0][0] for x in inter_terms])
print(len(pt_idx))

# Subgraph from terms to predict
sub_go_by_go = go_by_go[np.ix_(pt_idx,pt_idx)].copy()
sg2g_edg = np.transpose(np.nonzero(np.transpose(sub_go_by_go))).tolist()
sg2g = nx.DiGraph()
sg2g.add_nodes_from(np.arange(len(pt_idx)))
sg2g.add_edges_from(sg2g_edg)
print('GO subgraph (pred terms): nodes {0}, edges {1}'.format(sg2g.number_of_nodes(), sg2g.number_of_edges()))
print('Number of weakly conn. components: {}'.format(nx.number_weakly_connected_components(sg2g)))

# find possible root terms in go subgraph
proot_idx = list() # possible hierarchy roots
for i in range(len(pt_idx)):
  if np.count_nonzero(sub_go_by_go[i,:]) == 0: # terms wo ancestors
    proot_idx.append(i)
proot_idx = np.array(proot_idx)
print('Number of roots in GO subgraph: {0}'.format(len(proot_idx)))

# convert a bfs object to a list
def nodes_in_bfs(bfs, root):
  nodes = sorted(list(set([u for u,v in bfs] + [v for u,v in bfs])))
  nodes = np.setdiff1d(nodes, [root]).tolist()
  nodes = [root] + nodes
  return nodes

# detect isolated terms and create sub-hierarchies
hpt = list() # terms to predict and all terms in hierarchy
hroot_idx = list()
for root in proot_idx:
  bfs = nx.bfs_tree(sg2g, root).edges()

  if len(bfs) > 0: # if no isolated term
    hroot_idx.append(pt_idx[root])
    hpt.append(pt_idx[nodes_in_bfs(bfs, root)])

hroot_idx = np.array(hroot_idx)
len_hpt = [len(x) for x in hpt]
print('Number of isolated terms: {0}'.format(len(proot_idx)-len(hroot_idx)))
print('Number of sub-hierarchies: {0}'.format(len(hroot_idx)))
print('Average terms in sub-hierarchies: {0:.2f} [{1}-{2}]'.format(
  np.mean(len_hpt),
  np.min(len_hpt),
  np.max(len_hpt)))

# %%

"""
**Osa**
GO subgraph (pred terms): nodes 1840, edges 3205
Number of weakly conn. components: 12
Number of roots in GO subgraph: 26
Number of isolated terms: 10
Number of sub-hierarchies: 16
Average terms in sub-hierarchies: 176.50 [2-1017]

**Ath**
GO subgraph (pred terms): nodes 1840, edges 3205
Number of weakly conn. components: 12
Number of roots in GO subgraph: 26
Number of isolated terms: 10
Number of sub-hierarchies: 16
Average terms in sub-hierarchies: 176.50 [2-1017]

"""

# %%

# list sub-hierarchies
df_subh = pd.DataFrame(columns=['Root_idx','Root','Terms','Genes','Desc','Level'])
for i, rid in enumerate(hroot_idx):
  root = terms[rid]
  data = [rid, root, len(hpt[i])] # number of terms to predict in sub-hier.
  data += [np.count_nonzero(gene_by_go[:,rid])] # number of genes in sub.
  data += [term_def[term_def.Term==root].Desc.tolist()[0], g[root].level]
  df_subh.loc[i] = data

df_subh = df_subh.sort_values(by=['Terms','Genes'], ascending=False).reset_index(drop=True)
df_subh.to_csv('/home/miguel/projects/omics/transfer_data/zma_data/data/subhierarchies.csv', index=False)

# %%

# sub-hierarchies used for prediction
test_df_subh = df_subh[df_subh.Terms >= 9].sort_values(by=['Terms','Genes'], ascending=True).reset_index(drop=True)
print(test_df_subh)
test_r = test_df_subh.Root.tolist()
test_rid = test_df_subh.Root_idx.tolist()

test_hpt = list()
for i, root in enumerate(test_rid):
  idx = np.where(hroot_idx==root)[0][0]
  test_hpt.append(hpt[idx])

# %%
"""
  idx	Root	    Terms	Genes	Desc	                                        Level
 2988	GO:0040007	  9	  488	growth	                                          1
  220	GO:0002376	 15	  750	immune system process	                            1
 3366	GO:0044419	 26	  997	biological process involved in interspecies in...	1
 2593	GO:0032501	 33	 2463	multicellular organismal process	                1
 2284	GO:0022414	 52	 1461	reproductive process                              1
 2594	GO:0032502	103	 2764	developmental process                             1
 3969	GO:0050896	179	 5249	response to stimulus                              1
 4027	GO:0051179	185	 2308	localization	                                    1
 4383	GO:0065007	446	 4824	biological regulation	                            1
  903	GO:0008152  739	 9688	metabolic process	                                1
 1397	GO:0009987 1017	11335	cellular process	                                1
"""
# %%

def list2file(l, name):
  file = open(name, 'w')
  file.write('\n'.join([str(x) for x in l]))
  file.close()

def create_path(path):
  try: os.makedirs(path)
  except: pass

list2file(test_r, "/home/miguel/projects/omics/transfer_data/zma_data/data/roots.txt")
for x, root in zip(test_hpt, test_r):
  path = "/home/miguel/projects/omics/transfer_data/zma_data/data/{0}".format(root.replace(':',''))
  create_path(path)
  list2file(terms[x], "{0}/terms.txt".format(path))

# %%

for j, (x, root) in tqdm(enumerate(zip(test_hpt, test_rid))):
  hgenes = np.nonzero(gene_by_go[:,root])[0]
  hterms = terms[x] # terms to predict in hierarchy

  # Conver DAG to tree, will be used for prediction
  tree = mst(hgenes, x, gene_by_go.copy(), go_by_go.copy())
  hg2g = np.zeros((len(hterms),len(hterms)))
  for i, idx in enumerate(x):
    parents = direct_pa(idx, x, tree)
    parents = [np.where(x == p)[0][0] for p in parents]
    hg2g[i, parents] = 1
  hg2go = gene_by_go[hgenes,:].copy()

  q = deque()
  q.append((0,0)) # parent, level
  parents, level = 0, 0

  lcn = list()
  lcpn = list()
  levels = list()
  lcl, lastl, pterms, cterms = list(), 0, list(), list()

  while len(q) > 0:
    pos, l = q.popleft()
    children = np.nonzero(hg2g[:,pos])[0]

    # lcl order of prediction
    if lastl != l:
      lastl = l
      lcl.append("{0}= {1}".format(','.join(pterms), ','.join(cterms)))
      levels.append("{0}".format(sum([hg2go[:,idA[t]].sum() for t in cterms])))
      pterms, cterms = list(), list()
    pterms.append(hterms[pos])
    cterms += list(hterms[children])

    if len(children) > 0: # is a parent
      lcpn.append(("{0}= {1}".format(hterms[pos], ','.join(hterms[children])))) # save lcpn order of prediction

      parents += 1
      for c in children:
        lcn.append(("{0}= {1}".format(hterms[pos], hterms[c]))) # save lcn order of prediction
        q.append((c,l+1))

    level = max(level, l)

  path = "/home/miguel/projects/omics/transfer_data/zma_data/data/{0}".format(hterms[0].replace(':',''))
  create_path(path)
  list2file(lcn, "{0}/lcn.txt".format(path))
  list2file(lcpn, "{0}/lcpn.txt".format(path))
  list2file(lcl, "{0}/lcl.txt".format(path))
  list2file(levels, "{0}/levels.txt".format(path))

sys.exit()

# %%

def neighborhood_information(hgene, t):
  ans = list()
  for gid in hgene:
    neighbors = np.nonzero(gcn[:,gid])[0]
    ntotal = len(neighbors)
    nassc = np.count_nonzero(gene_by_go[neighbors,t])
    ans.append(nassc/ntotal) # proba. of being associated to class from neigh.
  return ans

# Scale data
def scale_data(data):
  # MinMaxScaler does not modify the distribution of data
  minmax_scaler = MinMaxScaler() # Must be first option

  new_data = pd.DataFrame()
  for fn in data.columns:
    scaled_feature = minmax_scaler.fit_transform(data[fn].values.reshape(-1,1))
    new_data[fn] = scaled_feature[:,0].tolist()

  return new_data

genes = np.array(genes)
# compute graph properties and feature embedding for each sub-hierarchy
for i, (root, hterms) in tqdm(enumerate(zip(test_rid, test_hpt)), total=len(test_rid), ascii="-#", desc="ZMA"):

  term = terms[root]

  print()
  print('#####################')
  print('Root term: {0}'.format(term))
  path = "/home/miguel/projects/omics/transfer_data/zma_data/data/{0}".format(term.replace(':',''))

  hgenes = np.nonzero(gene_by_go[:,root])[0]
  sgcn_adj = gcn[np.ix_(hgenes,hgenes)].copy() # create sub matrix terms_hier_idx hierarchy
  sgcn_edgelist = [(x,y,sgcn_adj[x,y]) for x,y in np.transpose(np.nonzero(sgcn_adj)).tolist()]
  sgcn = nx.Graph() # create graph for gcn (hierarchy gcn)
  sgcn.add_nodes_from(np.arange(len(hgenes)))
  sgcn.add_weighted_edges_from(sgcn_edgelist)

  gcc = sorted(nx.connected_components(sgcn), key=len, reverse=True)
  sgcn = sgcn.subgraph(gcc[0])
  nx.write_weighted_edgelist(sgcn, "{0}/gcn.edg".format(path), delimiter='\t')
  hgenes = hgenes[sgcn.nodes]

  list2file(genes[hgenes], "{0}/genes.txt".format(path))

  prob_df = pd.DataFrame()
  for  c in hterms:
    prb_feat = neighborhood_information(hgenes, c)
    prob_df["{0}-neigh".format(terms[c])] = prb_feat
  prob_df.to_csv('{0}/probs.csv'.format(path), index=False)

  sm_gene_by_go = gene_by_go[np.ix_(hgenes,hterms)].copy()

  # node embedding for prediction
  dimensions = len(hterms)
  p, q = 1, 0.5

  ss = time()
  n2v = pecanpy.DenseOTF(p=p, q=q, workers=8, verbose=False)
  n2v.read_edg("{0}/gcn.edg".format(path), weighted=True, directed=False)
  # embeddings = n2v.embed(dim=dimensions, num_walks=300, walk_length=5, window_size=5, epochs=1, verbose=False)
  embeddings = n2v.embed(dim=dimensions, num_walks=10, walk_length=80, window_size=10, epochs=1, verbose=False)

  # dimensionality reduction for clustering
  tsne = TSNE(n_components=2, random_state=7, perplexity=15)
  embeddings_2d = tsne.fit_transform(embeddings)

  clustering_model = AffinityPropagation(damping=0.9, random_state=202110)
  clustering_model.fit(embeddings_2d)
  yhat = clustering_model.predict(embeddings_2d)

  sh_df = pd.DataFrame()
  for i in range(dimensions):
    sh_df['emb_{0}'.format(i)] = pd.Series(embeddings[:,i])
  sh_df['emb_clust'] = pd.Series(yhat)
  sh_df = scale_data(sh_df)
  sh_df.to_csv('{0}/embedding_pn.csv'.format(path), index=False)

  # igraph
  sgcn_adj = gcn[np.ix_(hgenes,hgenes)].copy() # create sub matrix terms_hier_idx hierarchy
  sgcn_edgelist = [(x,y,sgcn_adj[x,y]) for x,y in np.transpose(np.nonzero(sgcn_adj)).tolist()]
  sgcn = ig.Graph.TupleList(sgcn_edgelist, weights=True)
  sgcn.to_undirected()
  sgcn.simplify(combine_edges="max")
  # ig.summary(sgcn)
  assert sgcn.is_connected() and sgcn.is_weighted() and sgcn.is_simple() and not sgcn.is_directed()

  # get node properties form graph
  deg = np.array(sgcn.degree())
  strength = np.array(sgcn.strength(sgcn.vs, weights="weight"))
  eccec = np.array(sgcn.eccentricity())
  auths = np.array(sgcn.authority_score(weights="weight"))
  hubs = np.array(sgcn.hub_score(weights="weight"))
  centr_betw = np.array(sgcn.betweenness(directed=False, weights="weight"))
  centr_clos = np.array(sgcn.closeness(weights="weight"))
  coren = np.array(sgcn.coreness())
  clust = np.array(sgcn.transitivity_local_undirected(mode="zero", weights="weight"))
  neigh_deg = np.array(sgcn.knn(weights="weight")[0])

  # add node properties to new df
  # cretae dataset
  sh_df = pd.DataFrame()
  sh_df["deg"] = pd.Series(deg) # degree
  sh_df["strength"] = pd.Series(strength) # strength, sum of the weights of all incident edges
  sh_df["eccec"] = pd.Series(eccec) # eccentricity
  sh_df["auths"] = pd.Series(auths) # authority score
  sh_df["hubs"] = pd.Series(hubs) # hub score
  sh_df["centr_betw"] = pd.Series(centr_betw) # betweenness centrality
  sh_df["centr_clos"] = pd.Series(centr_clos) # closeness centrality
  sh_df["coren"] = pd.Series(coren) # coreness
  sh_df["clust"] = pd.Series(clust) # clustering coeficcient
  sh_df["neigh_deg"] = pd.Series(neigh_deg) # average neighbor degree
  sh_df = scale_data(sh_df)
  sh_df.to_csv('{0}/struc.csv'.format(path), index=False)

  sh_df = pd.DataFrame()
  for i, trm in enumerate(terms[hterms]):
    if trm == term: continue
    sh_df[trm] = pd.Series(sm_gene_by_go[:,i].copy())
  sh_df.to_csv('{0}/labels.csv'.format(path), index=False)

  print('Terms:{0:6}\tGenes:{1:6}\tTime:{2:4.3f}s'.format(len(hterms), len(hgenes), time()-ss))

  # break
