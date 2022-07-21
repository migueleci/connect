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


org1 = sys.argv[1] # organism
org2 = sys.argv[2] # organism

root_list = readListFile("osa/data/roots.txt")
tresults = pd.DataFrame(columns=["Root","prc_micro","prc_macro","prc_macrow","pre_micro","pre_macro","pre_macrow","rec_micro","rec_macro","rec_macrow"])
seed = int(sys.argv[3]) # seed

df_best_source = pd.DataFrame(columns=["Root",org,org1,org2)
perf = np.zeros((len(root_list),4,9)) # datasets, organisms, measures
rprecis = pd.DataFrame(columns=["Root","tra","raw"]+[org1,org2])


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


  osa_path = "osa/data/{0}".format(root.replace(':',''))
  opath = "osa/preds/{0}/trans_{1}_{2}_{3}".format(root.replace(':',''), org1, org2, seed)
  create_path(opath)

  labels_order = readListFile("{0}/terms.txt".format(osa_path))[1:]
  genes = readListFile("{0}/genes.txt".format(osa_path)) # Genes for co-training/pu-learning, genes in old and new data

  labels = pd.read_csv("{0}/labels.csv".format(osa_path)) # labels - true gene-function associations
  data = pd.read_csv("{0}/embedding_pn.csv".format(osa_path)) # node embeddings of the GCN subgraph for subhierarchy
  props = pd.read_csv("{0}/struc.csv".format(osa_path)) # structural properties of the GCN subgraph for subhierarchy
  probs = pd.read_csv("{0}/probs.csv".format(osa_path)) # ratio of gene neighbors asscoeiated to each function
  data = pd.concat([data, props, probs], axis=1)

  # load both models and evaluation class
  estimator = RandomForestClassifier(n_estimators=200, min_samples_split=5, n_jobs=-1, random_state=seed) # Random forest classifier
  e = Evaluate()

  # Predicttions for rice using org data
  org1_pred = transfer_add_org(data, org1, root, estimator)
  org2_pred = transfer_add_org(data, org2, root, estimator)

  """ k-fold, the global approach is applied for each fold independently and the mean of the performance for each fold is the result """
  # osa_pred = np.zeros((len(test_index), len(labels_order))) # results for random forest multilabel classifier
  N = 5
  fold_idx = 1
  best_source = np.zeros(3)
  ds_perf = np.zeros((4,N,9)) # organisms, folds, measures
  pred_all = np.zeros(labels.shape)
  raw_pred_all = np.zeros(labels.shape)
  kfold = KFold(n_splits=N, shuffle=True, random_state=seed) # set the same folds for each model
  for Train_index, test_index in tqdm(kfold.split(labels), total=N, ascii="-#", desc="External CV"): # set the same folds for each model

    # prediction for the current fold (each fold is independent)
    osa_pred = np.zeros((len(test_index), len(labels_order))) # results for random forest multilabel classifier
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

      val_org1_pred = org1_pred.loc[val_index]
      val_org2_pred = org2_pred.loc[val_index]

      # compute alpha and beta for val fold
      nearest = np.zeros(y_val.shape)
      for i in range(y_val.shape[0]):
        true = y_val.values[i,:]
        rpred = val_pred[i,:]
        o1pred = val_org1_pred.values[i,:]
        o2pred = val_org2_pred.values[i,:]

        preds = np.array([rpred,o1pred,o2pred]).T
        for j in range(len(true)):
          nearest[i,j] = np.abs(preds[j,:] - true[j]).argmin()

      tnearest = np.zeros((3,y_val.shape[1]))
      trans_pred = np.zeros(y_val.shape)
      for j in range(y_val.shape[1]):
        count = Counter(nearest[:,j])
        for c in count:
          tnearest[int(c),j] = count[c]
        assert tnearest[:,j].sum() == y_val.shape[0]

        a, b, c = tnearest[:,j]/y_val.shape[0]
        trans_pred[:,j] = a * val_pred[:,j] + b * val_org1_pred.values[:,j] + c * val_org2_pred.values[:,j]

      tnearest = tnearest / y_val.shape[0]
      constants.append(tnearest)

      tmes.append(e.multiclass_classification_measures(trans_pred, y_val))

    # constants = constants[np.array(tmes).T[5,:].argmax()]  # take the constants with higher pooled auprc
    constants = np.array(constants).mean(axis=0) # take the average of the constants

    X_train, X_test = data.loc[Train_index], data.loc[test_index] # train-test split for random forest
    y_train, y_test = labels.loc[Train_index], labels.loc[test_index] # train-test split of y

    estimator.fit(X_train, y_train)
    test_pred = np.zeros(y_test.shape) # results for random forest multilabel classifier
    _pred = estimator.predict_proba(X_test)
    for cidx, x, cls in zip(range(len(labels.columns)), _pred, estimator.classes_):
      test_pred[:,cidx] = 1 - x[:,0].copy() if cls[0] == 0 else x[:,0].copy()


    test_org1_pred = org1_pred.loc[test_index]
    test_org2_pred = org2_pred.loc[test_index]
    trans_pred = np.zeros(y_test.shape)
    for j in range(y_test.shape[1]):
      a, b, c = constants[:,j]
      assert np.isclose(a + b + c, 1)
      trans_pred[:,j] = a * test_pred[:,j] + b * test_org1_pred.values[:,j] + c * test_org2_pred.values[:,j]

      # best = np.zeros(3)
      # best[constants[:,j].argmax()] = 1
      # a, b, c = best
      # trans_pred[:,j] = a * test_pred[:,j] + b * test_org1_pred.values[:,j] + c * test_org2_pred.values[:,j]

    """ storing predictions for the current fold """
    pred_df = pd.DataFrame(trans_pred, columns=labels_order)
    pred_df.index = test_index
    pred_df.to_csv("{0}/{1}.csv".format(opath, fold_idx))

    pred_df = pd.DataFrame(test_pred, columns=labels_order)
    pred_df.index = test_index
    pred_df.to_csv("{0}/{1}.csv".format(rpath, fold_idx))
    fold_idx += 1


    """ compute the performance of the whole hierarchy """
    tmes = e.multiclass_classification_measures(trans_pred, y_test)
    rmes = e.multiclass_classification_measures(test_pred, y_test)
    o1mes = e.multiclass_classification_measures(test_org1_pred.values, y_test)
    o2mes = e.multiclass_classification_measures(test_org2_pred.values, y_test)

    ds_perf[0,fold_idx-2,:] = tmes[1:10] # transfer performance
    ds_perf[1,fold_idx-2,:] = rmes[1:10] # rice performance
    ds_perf[2,fold_idx-2,:] = o1mes[1:10] # organism 1 performance
    ds_perf[3,fold_idx-2,:] = o2mes[1:10] # organism 2 performance


  """ performance of the hierarchy is computed, mean between folds """
  pprint(ds_perf[0,:,:].mean(axis=0)[5], ds_perf[0,:,:].mean(axis=0)[3], ds_perf[0,:,:].mean(axis=0)[4], 'tra')
  pprint(ds_perf[1,:,:].mean(axis=0)[5], ds_perf[1,:,:].mean(axis=0)[3], ds_perf[1,:,:].mean(axis=0)[4], 'osa')
  pprint(ds_perf[2,:,:].mean(axis=0)[5], ds_perf[2,:,:].mean(axis=0)[3], ds_perf[2,:,:].mean(axis=0)[4], org1)
  pprint(ds_perf[3,:,:].mean(axis=0)[5], ds_perf[3,:,:].mean(axis=0)[3], ds_perf[3,:,:].mean(axis=0)[4], org2)

  perf[ridx,0,:] = ds_perf[0,:,:].mean(axis=0)
  perf[ridx,1,:] = ds_perf[1,:,:].mean(axis=0)
  perf[ridx,2,:] = ds_perf[2,:,:].mean(axis=0)
  perf[ridx,3,:] = ds_perf[3,:,:].mean(axis=0)

  tresults.loc[ridx] = [root] + list(perf[ridx,0,:])
  tresults.to_csv("osa/preds/trans_{0}_{1}_avg.csv".format(org1, org2), index=False)

  rresults.loc[ridx] = [root] + list(perf[ridx,1,:])
  rresults.to_csv("osa/preds/rice_{0}_{1}_avg.csv".format(org1, org2), index=False)


  org1n = 'Arabidopsis' if org1 == 'ath' else 'Soybean' if org1 == 'gly' else 'Maize'
  org2n = 'Arabidopsis' if org2 == 'ath' else 'Soybean' if org2 == 'gly' else 'Maize'
  labels = ['Transfer','Rice',org1n,org2n]
  markers = ['--P','--o','--s','--s']
  xticks = root_list[:ridx+1]

  # Random forest
  create_path('figs/trans_{0}_{1}'.format(org1, org2))
  with PdfPages('figs/trans_{0}_{1}/prc_avg.pdf'.format(org1, org2)) as pdf:
    dmax = np.nanmax(perf[:ridx+1,:,2])*1.05
    line_plot(perf[:ridx+1,:,2].T, xticks, labels, markers, 'Sub-hierarchy', 'pooled AUPRC (micro)', ylim=[0.2,dmax], multipdf=pdf)

    dmax = np.nanmax(perf[:ridx+1,:,0])*1.05
    line_plot(perf[:ridx+1,:,0].T, xticks, labels, markers, 'Sub-hierarchy', 'AUPRC (macro)', ylim=[0.0,dmax], multipdf=pdf)

    dmax = np.nanmax(perf[:ridx+1,:,1])*1.05
    line_plot(perf[:ridx+1,:,1].T, xticks, labels, markers, 'Sub-hierarchy', 'AUPRC (macro weighted)', ylim=[0.1,dmax], multipdf=pdf)

  with PdfPages('figs/trans_{0}_{1}/pre_avg.pdf'.format(org1, org2)) as pdf:
    dmax = np.nanmax(perf[:ridx+1,:,5])*1.05
    line_plot(perf[:ridx+1,:,5].T, xticks, labels, markers, 'Sub-hierarchy', 'pooled AUPREC (micro)', ylim=[0.2,dmax], multipdf=pdf)

    dmax = np.nanmax(perf[:ridx+1,:,3])*1.05
    line_plot(perf[:ridx+1,:,3].T, xticks, labels, markers, 'Sub-hierarchy', 'AUPREC (macro)', ylim=[0.0,dmax], multipdf=pdf)

    dmax = np.nanmax(perf[:ridx+1,:,4])*1.05
    line_plot(perf[:ridx+1,:,4].T, xticks, labels, markers, 'Sub-hierarchy', 'AUPREC (macro weighted)', ylim=[0.1,dmax], multipdf=pdf)

  with PdfPages('figs/trans_{0}_{1}/rec_avg.pdf'.format(org1, org2)) as pdf:
    dmax = np.nanmax(perf[:ridx+1,:,8])*1.05
    line_plot(perf[:ridx+1,:,8].T, xticks, labels, markers, 'Sub-hierarchy', 'pooled AURECC (micro)', ylim=[0.2,dmax], multipdf=pdf)

    dmax = np.nanmax(perf[:ridx+1,:,6])*1.05
    line_plot(perf[:ridx+1,:,6].T, xticks, labels, markers, 'Sub-hierarchy', 'AURECC (macro)', ylim=[0.0,dmax], multipdf=pdf)

    dmax = np.nanmax(perf[:ridx+1,:,7])*1.05
    line_plot(perf[:ridx+1,:,7].T, xticks, labels, markers, 'Sub-hierarchy', 'AURECC (macro weighted)', ylim=[0.1,dmax], multipdf=pdf)

  # break
