import os
import sys

import numpy as np
import pandas as pd
from collections import Counter

from tqdm import tqdm, tnrange
from time import time

import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from matplotlib import rc
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from evaluate import *

rc('font', family='serif', size=18)
rc('text', usetex=False)

COLORS = ['#1f77b4', '#ff7f0e', '#8c564b', '#2ca02c', '#9467bd', '#d62728',
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=3)


def create_path(path):
  try: os.makedirs(path)
  except: pass

# read a txt file containing a list, one item per line
def readListFile(filename):
  file = open(filename, 'r')
  tmp = [x.strip() for x in file.readlines()]
  file.close()
  return np.array(tmp)


# pretty print of results for a model
def pprint(pooled, macro, macrow, name=None):
    print("#{0} pooled:{1:2.4f}\t\tmacro:{2:2.4f}\t\tmacrow:{3:2.4f}".format(
      '' if name is None else ' {0}:'.format(name),
      pooled, macro, macrow, time)
    )


def line_plot(data,xticks,labels, markers,xlabel,ylabel,title=None,fname=None,path=None,ylim=None,multipdf=None):
  fig, ax = plt.subplots(figsize=(10,10))
  x = np.arange(len(xticks))
  for idx, (y, l, m) in enumerate(zip(data,labels, markers)):
    plt.plot(x, y, m, color=COLORS[idx], label=l)
  if ylim is not None:
    plt.ylim(ylim)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  if title is not None:
    plt.title(title)
  plt.xticks(x, xticks, rotation=90)
  plt.grid(axis='both', linestyle='--')
  if len(labels) > 1:
    plt.legend(loc='best')
  plt.tight_layout()
  if multipdf is None:
    plt.savefig('{0}/{1}.pdf'.format(path, fname), format='pdf', dpi=600)
  else:
    multipdf.savefig(dpi=600)
  plt.close()

org = sys.argv[1] # seed
root_list = readListFile("{0}/data/roots.txt".format(org))
root_list = ["GO:0051179","GO:0008152"]
results = pd.DataFrame(columns=["Root","prc_micro","prc_macro","prc_macrow","pre_micro","pre_macro","pre_macrow","rec_micro","rec_macro","rec_macrow"])
seed = int(sys.argv[2]) # seed

add_orgs = [x for x in ["ath","gly","osa","zma"] if x != org]
add_orgs = [x for x in ["osa","zma"] if x != org]
df_best_source = pd.DataFrame(columns=["Root"]+[org]+add_orgs)
perf = np.zeros((len(root_list),len(add_orgs)+2,9)) # datasets, organisms, measures
rprecis = pd.DataFrame(columns=["Root","tra","raw"]+add_orgs)


def transfer_add_org(X_test, source_name, root, estimator):
  source_path = "{0}/data/{1}".format(source_name, root.replace(':',''))
  source_labels = pd.read_csv("{0}/labels.csv".format(source_path)) # labels - true gene-function associations
  source_data = pd.read_csv("{0}/embedding_pn.csv".format(source_path)) # node embeddings of the GCN subgraph for subhierarchy
  source_props = pd.read_csv("{0}/struc.csv".format(source_path)) # structural properties of the GCN subgraph for subhierarchy
  source_probs = pd.read_csv("{0}/probs.csv".format(source_path)) # ratio of gene neighbors asscoeiated to each function
  source_data = pd.concat([source_data, source_props, source_probs], axis=1)

  X_train, y_train = source_data.copy(), source_labels.copy() # train-test split of y

  """ training and prediction  """
  estimator.fit(X_train, y_train)
  _pred = estimator.predict_proba(X_test)
  pred = np.zeros((X_test.shape[0],y_train.shape[1]))
  for cidx, x, cls in zip(range(len(source_labels.columns)), _pred, estimator.classes_):
    pred[:,cidx] = 1 - x[:,0].copy() if cls[0] == 0 else x[:,0].copy()
  return pred


for ridx, root in enumerate(root_list):
  if ridx>0: print()
  print("{0}. Root: {1}".format(ridx+1, root))
  print("{0}------------------".format('--' if ridx>8 else '-'))


  org_path = "{0}/data/{1}".format(org, root.replace(':',''))
  out_path = "{0}/preds/{1}/trans_all_{2}".format(org, root.replace(':',''), seed)
  create_path(out_path)

  labels_order = readListFile("{0}/terms.txt".format(org_path))[1:]
  genes = readListFile("{0}/genes.txt".format(org_path)) # Genes for co-training/pu-learning, genes in old and new data

  labels = pd.read_csv("{0}/labels.csv".format(org_path)) # labels - true gene-function associations
  data = pd.read_csv("{0}/embedding_pn.csv".format(org_path)) # node embeddings of the GCN subgraph for subhierarchy
  props = pd.read_csv("{0}/struc.csv".format(org_path)) # structural properties of the GCN subgraph for subhierarchy
  probs = pd.read_csv("{0}/probs.csv".format(org_path)) # ratio of gene neighbors asscoeiated to each function
  data = pd.concat([data, props, probs], axis=1)

  lcn = open("osa/data/{0}/lcn.txt".format(root.replace(':','')), 'r') # load lcn order
  lcn = [x.strip() for x in lcn.readlines()]
  parent_map = dict([(l.split('=')[1].strip(),l.split('=')[0].strip()) for l in lcn])

  # load both models and evaluation class
  estimator = RandomForestClassifier(n_estimators=200, min_samples_split=5, n_jobs=-1, random_state=seed) # Random forest classifier
  e = Evaluate()

  # Predicttions for rice using org data
  add_preds = [transfer_add_org(data, x, root, estimator) for x in add_orgs]

  """ k-fold, the global approach is applied for each fold independently and the mean of the performance for each fold is the result """
  N = 5
  fold_idx = 1
  best_source = np.zeros(len(add_orgs)+1)
  ds_perf = np.zeros((len(add_orgs)+2,N,9)) # organisms, folds, measures
  pred_all = np.zeros(labels.shape)
  raw_pred_all = np.zeros(labels.shape)
  kfold = KFold(n_splits=N, shuffle=True, random_state=seed) # set the same folds for each model
  for Train_index, test_index in tqdm(kfold.split(labels), total=N, ascii="-#", desc="External CV"): # set the same folds for each model

    # prediction for the current fold (each fold is independent)
    X_test, y_test = data.loc[test_index], labels.loc[test_index] # train-test split for random forest
    constants = list()
    tmes = list()

    ikfold = KFold(n_splits=3, shuffle=True, random_state=seed) # set the same folds for each model
    for itrain_index, ival_index in ikfold.split(Train_index): # tqdm(ikfold.split(Train_index), total=3, ascii="-#", desc="Internal CV"): # set the same folds for each model
      train_index = Train_index[itrain_index]
      val_index = Train_index[ival_index]

      """ create y and x dataset for the current hierarchy """
      X_train, X_val = data.loc[train_index], data.loc[val_index] # train-test split for random forest
      y_train, y_val = labels.loc[train_index], labels.loc[val_index] # train-test split of y

      """ training and prediction """
      estimator.fit(X_train, y_train)
      val_pred = np.zeros(y_val.shape) # results for random forest multilabel classifier
      _pred = estimator.predict_proba(X_val)
      for cidx, x, cls in zip(range(len(labels.columns)), _pred, estimator.classes_):
        val_pred[:,cidx] = 1 - x[:,0].copy() if cls[0] == 0 else x[:,0].copy()

      add_val_pred = [x[val_index,:].copy() for x in add_preds]

      # compute alpha and beta for val fold
      nearest = np.zeros(y_val.shape)
      for i in range(y_val.shape[0]):
        true = y_val.values[i,:]
        ipred = val_pred[i,:]
        add_ipred = [x[i,:] for x in add_val_pred]

        preds = np.array([ipred]+add_ipred).T
        for j in range(len(true)):
          nearest[i,j] = np.abs(preds[j,:] - true[j]).argmin()

      tnearest = np.zeros((len(add_orgs)+1,y_val.shape[1]))
      trans_pred = np.zeros(y_val.shape)
      for j in range(y_val.shape[1]):
        count = Counter(nearest[:,j])
        for c in count:
          tnearest[int(c),j] = count[c]
        assert tnearest[:,j].sum() == y_val.shape[0]

        # alpha = tnearest[:,j]/y_val.shape[0]
        # trans_pred[:,j] = alpha[0] * val_pred[:,j]
        # for a, x in zip(alpha[1:], add_val_pred):
        #   trans_pred[:,j] += a * x[:,j]

        # best = np.zeros(len(add_orgs)+1)
        # best[alpha.argmax()] = 1
        # trans_pred[:,j] = best[0] * val_pred[:,j]
        # for a, x in zip(best[1:], add_val_pred):
        #   trans_pred[:,j] += a * x[:,j]

      tnearest = tnearest / y_val.shape[0]
      constants.append(tnearest)

      # tmes.append(e.multiclass_classification_measures(trans_pred, y_val))

    # constants = constants[np.array(tmes).T[3,:].argmax()]  # take the constants with higher pooled auprc
    constants = np.array(constants).mean(axis=0) # take the average of the constants
    df_const = pd.DataFrame(constants, columns=labels.columns)
    df_const['org'] = [org]+add_orgs
    # df_const.to_csv("{0}/const_{1}.csv".format(out_path, fold_idx))

    X_train, X_test = data.loc[Train_index], data.loc[test_index] # train-test split for random forest
    y_train, y_test = labels.loc[Train_index], labels.loc[test_index] # train-test split of y

    estimator.fit(X_train, y_train)
    test_pred = np.zeros(y_test.shape) # results for random forest multilabel classifier
    _pred = estimator.predict_proba(X_test)
    for cidx, x, cls in zip(range(len(labels.columns)), _pred, estimator.classes_):
      test_pred[:,cidx] = 1 - x[:,0].copy() if cls[0] == 0 else x[:,0].copy()

    add_test_pred = [x[test_index,:].copy() for x in add_preds]
    trans_pred = np.zeros(y_test.shape)
    for j in range(y_test.shape[1]):
      alpha = constants[:,j]
      assert np.isclose(alpha.sum(), 1)
      trans_pred[:,j] = alpha[0] * test_pred[:,j]
      for a, x in zip(alpha[1:], add_test_pred):
        trans_pred[:,j] += a * x[:,j]

      best = np.zeros(len(add_orgs)+1)
      best[constants[:,j].argmax()] = 1
      alpha = best
      trans_pred[:,j] = alpha[0] * test_pred[:,j]
      for a, x in zip(alpha[1:], add_test_pred):
        trans_pred[:,j] += a * x[:,j]

      best_source += best

    """ hierarchy constraint """
    assert len(labels_order) == y_test.shape[1]
    for j in range(y_test.shape[1]):
      parent = parent_map[labels_order[j]]
      if parent != root:
        idx_parent = np.where(labels_order==parent)[0][0] # get the indices of parent
        trans_pred[:,j] = np.minimum(trans_pred[:,j], trans_pred[:,idx_parent]) # compute cumprob for random forest

    """ storing predictions for the current fold """
    pred_df = pd.DataFrame(trans_pred, columns=labels_order)
    pred_df.index = test_index
    # pred_df.to_csv("{0}/{1}.csv".format(out_path, fold_idx))
    fold_idx += 1

    pred_all[test_index,:] = trans_pred.copy()
    raw_pred_all[test_index,:] = test_pred.copy()

    """ compute the performance of the whole hierarchy """
    tmes = e.multiclass_classification_measures(trans_pred, y_test) # org transfer pred`
    rmes = e.multiclass_classification_measures(test_pred, y_test) # org raw pred
    add_mes = [e.multiclass_classification_measures(x, y_test) for x in add_test_pred]

    ds_perf[0,fold_idx-2,:] = tmes[1:10] # transfer performance
    ds_perf[1,fold_idx-2,:] = rmes[1:10] # raw performance
    for i in range(len(add_orgs)):
      ds_perf[i+2,fold_idx-2,:] = add_mes[i][1:10] # additional orgs performance


  """ performance of the hierarchy is computed, mean between folds """
  pprint(ds_perf[0,:,:].mean(axis=0)[5], ds_perf[0,:,:].mean(axis=0)[3], ds_perf[0,:,:].mean(axis=0)[4], 'tra '+org)
  pprint(ds_perf[1,:,:].mean(axis=0)[5], ds_perf[1,:,:].mean(axis=0)[3], ds_perf[1,:,:].mean(axis=0)[4], 'raw '+org)
  for i in range(len(add_orgs)):
    pprint(ds_perf[i+2,:,:].mean(axis=0)[5], ds_perf[i+2,:,:].mean(axis=0)[3], ds_perf[i+2,:,:].mean(axis=0)[4], add_orgs[i])

  perf[ridx,0,:] = ds_perf[0,:,:].mean(axis=0)
  perf[ridx,1,:] = ds_perf[1,:,:].mean(axis=0)
  for i in range(len(add_orgs)):
    perf[ridx,i+2,:] = ds_perf[i+2,:,:].mean(axis=0)

  results.loc[ridx] = [root] + list(perf[ridx,0,:]) # transfer
  # results.to_csv("{0}/preds/trans_all_dumf_{1}.csv".format(org,seed), index=False)

  rprecis.loc[ridx] = [root] + list(perf[ridx,:,5])
  rprecis.to_csv("{0}/preds/trans_all_prec_v7_{1}.csv".format(org,seed), index=False)

  df_best_source.loc[ridx] = [root] + list(best_source)
  df_best_source.to_csv("{0}/preds/trans_all_count_v7_{1}.csv".format(org,seed), index=False)

  ndict = dict([('ath','Arabidopsis'),('gly','Soybean'),('osa','Rice'),('zma','Maize')])
  flabels = ['Transfer {0}'.format(ndict[org]),'Raw  {0}'.format(ndict[org])]+[ndict[x] for x in add_orgs]
  markers = ['--P','--o','--.','--.','--.']
  xticks = root_list[:ridx+1]

  create_path("figs/trans_all_{0}/auprec_{1}".format(org,seed))
  tmes = e.multiclass_classification_measures(pred_all, labels)
  rmes = e.multiclass_classification_measures(raw_pred_all, labels)
  add_mes = [e.multiclass_classification_measures(x, labels) for x in add_preds]
  xti = ['{0:.2f}'.format(ix) if j%2==0 else "" for j, ix in enumerate(np.linspace(0, 1, 51))]
  pdata = np.nan_to_num([tmes[10],rmes[10]] + [x[10] for x in add_mes])
  plabels = ['Transfer {0} AUC={1:.2f}'.format(ndict[org], tmes[6]),'Raw  {0} AUC={1:.2f}'.format(ndict[org], rmes[6])]+["{0} AUC={1:.2f}".format(ndict[add_orgs[i]], add_mes[i][6]) for i in range(len(add_orgs))]
  dmax = np.nanmax(pdata)*1.05
  line_plot(pdata, xti, plabels, markers, 'Sub-hierarchy', 'Pooled precision', ylim=[0,dmax], fname='{0}_v7'.format(root.replace(":","")),path="figs/trans_all_{0}/auprec_{1}".format(org,seed))

  # Random forest
  # create_path('figs/trans_all')
  # with PdfPages('figs/trans_all_{0}/prc_{1}.pdf'.format(org,seed)) as pdf:
  #   dmax = np.nanmax(perf[:ridx+1,:,2])*1.05
  #   line_plot(perf[:ridx+1,:,2].T, xticks, flabels, markers, 'Sub-hierarchy', 'pooled AUPRC (micro)', ylim=[0.2,dmax], multipdf=pdf)
  #
  #   dmax = np.nanmax(perf[:ridx+1,:,0])*1.05
  #   line_plot(perf[:ridx+1,:,0].T, xticks, flabels, markers, 'Sub-hierarchy', 'AUPRC (macro)', ylim=[0.0,dmax], multipdf=pdf)
  #
  #   dmax = np.nanmax(perf[:ridx+1,:,1])*1.05
  #   line_plot(perf[:ridx+1,:,1].T, xticks, flabels, markers, 'Sub-hierarchy', 'AUPRC (macro weighted)', ylim=[0.1,dmax], multipdf=pdf)
  #
  # with PdfPages('figs/trans_all_{0}/pre_{1}.pdf'.format(org,seed)) as pdf:
  #   dmax = np.nanmax(perf[:ridx+1,:,5])*1.05
  #   line_plot(perf[:ridx+1,:,5].T, xticks, flabels, markers, 'Sub-hierarchy', 'pooled AUPREC (micro)', ylim=[0.2,dmax], multipdf=pdf)
  #
  #   dmax = np.nanmax(perf[:ridx+1,:,3])*1.05
  #   line_plot(perf[:ridx+1,:,3].T, xticks, flabels, markers, 'Sub-hierarchy', 'AUPREC (macro)', ylim=[0.0,dmax], multipdf=pdf)
  #
  #   dmax = np.nanmax(perf[:ridx+1,:,4])*1.05
  #   line_plot(perf[:ridx+1,:,4].T, xticks, flabels, markers, 'Sub-hierarchy', 'AUPREC (macro weighted)', ylim=[0.1,dmax], multipdf=pdf)
  #
  # with PdfPages('figs/trans_all_{0}/rec_{1}.pdf'.format(org,seed)) as pdf:
  #   dmax = np.nanmax(perf[:ridx+1,:,8])*1.05
  #   line_plot(perf[:ridx+1,:,8].T, xticks, flabels, markers, 'Sub-hierarchy', 'pooled AURECC (micro)', ylim=[0.2,dmax], multipdf=pdf)
  #
  #   dmax = np.nanmax(perf[:ridx+1,:,6])*1.05
  #   line_plot(perf[:ridx+1,:,6].T, xticks, flabels, markers, 'Sub-hierarchy', 'AURECC (macro)', ylim=[0.0,dmax], multipdf=pdf)
  #
  #   dmax = np.nanmax(perf[:ridx+1,:,7])*1.05
  #   line_plot(perf[:ridx+1,:,7].T, xticks, flabels, markers, 'Sub-hierarchy', 'AURECC (macro weighted)', ylim=[0.1,dmax], multipdf=pdf)

  with PdfPages('figs/trans_all_{0}/pooled_v7_{1}.pdf'.format(org,seed)) as pdf:
    dmax = np.nanmax(perf[:ridx+1,:,2])*1.05
    line_plot(perf[:ridx+1,:,2].T, xticks, flabels, markers, 'Sub-hierarchy', 'pooled AUPRC (micro)', ylim=[0.2,dmax], multipdf=pdf)

    dmax = np.nanmax(perf[:ridx+1,:,5])*1.05
    line_plot(perf[:ridx+1,:,5].T, xticks, flabels, markers, 'Sub-hierarchy', 'pooled AUPREC (micro)', ylim=[0.2,dmax], multipdf=pdf)

    dmax = np.nanmax(perf[:ridx+1,:,8])*1.05
    line_plot(perf[:ridx+1,:,8].T, xticks, flabels, markers, 'Sub-hierarchy', 'pooled AURECC (micro)', ylim=[0.2,dmax], multipdf=pdf)

  print(best_source)
  # break
