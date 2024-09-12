# %%
from time import time
import numpy as np
import pandas as pd
from matplotlib.pylab import mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from xgboost import XGBRegressor
import pickle
import shap
from time import time
shap.initjs()

mpl.rcParams['font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus']=False 

start = time()

config = {
    'seed':1122,
    'data_dir':r'D:\Delta\OptML\Sim_2',
    'batch': 32,
    'epochs': 500,
    'patience':20,
    'visualize':False,
    'negative_slope':0.01,
    'udf_UPH':70,
    'MAE_threshold':0.95,
    'loss':{
        'train':[],
        'valid':[],
    }
}
df = pd.read_csv('uph_opt.csv', encoding='gbk').dropna()
# %%
train_X, val_X, train_y, val_y = train_test_split(
#     df.drop(columns=['Begin_Hour', 'LINE_NAME', 'MODEL_NAME', 'Date', 'UPH', 'UPH_Equv', 'MO_NUMBER', 'REAL_QTY_Equv']), 
    df.drop(columns=['UPH', 'UPH_Equv']), 
    df['UPH'], 
    train_size=0.8,
    random_state=config['seed'])

# %%
config['D_in'] = train_X.shape[1]
try: 
    config['D_out'] = train_y.shape[1]
except IndexError:
    config['D_out'] = 1
# %%

xgb = XGBRegressor(random_state=config['seed'])
xgb.fit(train_X, train_y)
pickle.dump(xgb, open('xgb_response_surface.pkl', "wb"))


def rmse_scorer(clf, X, y):
    y_pred = clf.predict(X)
    return {'rmse':np.mean(np.sqrt((y - y_pred) ** 2))}

cv_result = cross_validate(xgb, train_X, train_y, cv=5, scoring=rmse_scorer)
print(cv_result['test_rmse'])
y_pred = xgb.predict(val_X)
print('RMSE : ', np.mean(np.sqrt((val_y - y_pred) ** 2)))
# %%

explainer = shap.Explainer(xgb)
shap_values = explainer(train_X)

# visualize the first prediction's explanation
for i in range(5):
    fig = plt.figure()
    shap.plots.bar(shap_values[i], max_display=15, show=False)
    plt.xlabel('貢獻值')
    plt.show()

# %%
fig = plt.figure()
fig.patch.set_facecolor('white')
fig.patch.set_alpha(0)
shap.plots.beeswarm(shap_values, max_display=25, show=False)
plt.xlabel('貢獻值')

# %%
def MAE(l):
    return np.mean(np.abs(l - np.mean(l, axis=0)), axis=0)
mae_result = MAE(shap_values.values)
selected_train_X = train_X[
    train_X.columns[np.argsort(mae_result)[::-1]][:sum(
        np.cumsum(np.sort(mae_result)[::-1]/mae_result.sum()
                  ) < config['MAE_threshold'])+1]]

unselected_train_X = train_X[[c for c in train_X.columns if c not in selected_train_X.columns]]
unselected_mean = unselected_train_X.mean(axis=0)
# selected_val_X = val_X[val_X.columns[np.argsort(mae_result)[::-1]][:sum(np.cumsum(np.sort(mae_result)[::-1]/mae_result.sum()) < config['MAE_threshold'])+1]]
# %%
fig = plt.figure(figsize=(13, 7))
plt.bar(
    train_X.columns[np.argsort(mae_result)[::-1]][:10], 
    (np.sort(mae_result)[::-1]/mae_result.sum())[:10])
plt.title('MAE占比')
fig.tight_layout()
plt.show()
# %%
# selected_xgb = XGBRegressor(random_state=config['seed'])
# selected_xgb.fit(selected_train_X, train_y)
# pickle.dump(selected_xgb, open('selected_xgb.pkl', "wb"))

# y_pred = selected_xgb.predict(selected_val_X)
# print('RMSE : ', np.mean(np.sqrt((val_y - y_pred) ** 2)))
# %%
from sklearn.metrics import confusion_matrix

tree_clf = DecisionTreeClassifier(
    min_samples_leaf=50, 
    random_state=config['seed'])
tree_clf.fit(selected_train_X.values, (xgb.predict(train_X) >= config['udf_UPH']).astype(int).astype(str))

confusion_matrix(
    (xgb.predict(train_X) >= config['udf_UPH']).astype(int).astype(str), 
    tree_clf.predict(selected_train_X.values))

# %%
node_status = np.squeeze(tree_clf.tree_.value)
count_leaf = 0
flow = {}
def gini(status_list):
    return (1 - np.sum((status_list/status_list.sum())**2))//0.0001/10000

for i in range(tree_clf.tree_.node_count):
    if tree_clf.tree_.children_left[i] == tree_clf.tree_.children_right[i]:
        if i not in flow.keys():
            flow[i]={}
        flow[i]['0'] = node_status[i][0]
        flow[i]['1'] = node_status[i][1]
        flow[i]['confidence'] = node_status[i][1]/(node_status[i][1]+node_status[i][0])
        count_leaf += 1
        flow[i]['class'] = '0' if node_status[i][0] >= node_status[i][1] else '1'
        flow[i]['final_gini'] = gini(node_status[i])

    else:
        if tree_clf.tree_.children_left[i] not in flow.keys():
            flow[tree_clf.tree_.children_left[i]]={}
        if tree_clf.tree_.children_right[i] not in flow.keys():
            flow[tree_clf.tree_.children_right[i]]={}

        if tree_clf.tree_.threshold[i] < 0:
            flow[tree_clf.tree_.children_left[i]]['top'] = [
                i, 
                'lambda x: x[{feature}] + {threshold}'.format(
                    feature=tree_clf.tree_.feature[i], 
                    threshold=tree_clf.tree_.threshold[i]),
                gini(node_status[i])-gini(node_status[tree_clf.tree_.children_left[i]])
                ]
            # print('case 1')
            flow[tree_clf.tree_.children_right[i]]['top'] = [
                i, 
                'lambda x: -{threshold} - x[{feature}]'.format(
                    feature=tree_clf.tree_.feature[i], 
                    threshold=tree_clf.tree_.threshold[i]),
                gini(node_status[i])-gini(node_status[tree_clf.tree_.children_right[i]])
                ]
            # print('case 3')


        else:
            flow[tree_clf.tree_.children_left[i]]['top'] = [
                i, 
                'lambda x: x[{feature}] - {threshold}'.format(
                    feature=tree_clf.tree_.feature[i], 
                    threshold=tree_clf.tree_.threshold[i]),
                gini(node_status[i])-gini(node_status[tree_clf.tree_.children_left[i]])
                ]
            # print('case 2')

            flow[tree_clf.tree_.children_right[i]]['top'] = [
                i, 
                'lambda x: {threshold} - x[{feature}]'.format(
                    feature=tree_clf.tree_.feature[i], 
                    threshold=tree_clf.tree_.threshold[i]),
                gini(node_status[i])-gini(node_status[tree_clf.tree_.children_right[i]])
                ]
            # print('case 4')
        
# %%
root = {}
for k in flow.keys():
    for kk in flow[k].keys():
        if 'confidence' in kk:
            node_order = [str(k)]
            constraint_order = []
            node_order.append(str(flow[k]['top'][0]))
            constraint_order.append(str(flow[k]['top'][1]))

            top = flow[k]['top'][0]
            while top in flow.keys():
                node_order.append(str(flow[top]['top'][0]))
                constraint_order.append(str(flow[top]['top'][1]))
                top = flow[top]['top'][0]

            root[str(k)] = {
                'node_order':node_order[::-1],
                'constraint_order':constraint_order[::-1],
                'final_gini':str(flow[k]['final_gini']),
                'class':flow[k]['class']
            }
        else:
            continue

# %%
from matplotlib.colors import ListedColormap, to_rgb
colors = ['crimson', 'dodgerblue']

fig = plt.figure(dpi=200, figsize=(13, 7))
artists = tree.plot_tree(
    tree_clf, 
    feature_names=list(selected_train_X.columns),
    class_names=['限制空間', '可行解區'],
    label='none',
    fontsize=10,
    filled=True, 
    rounded=True)
for artist, impurity, value in zip(artists, tree_clf.tree_.impurity, tree_clf.tree_.value):
    r, g, b = to_rgb(colors[np.argmax(value)])
    f = impurity * 2
    artist.get_bbox_patch().set_facecolor((f + (1-f)*r, f + (1-f)*g, f + (1-f)*b))
    artist.get_bbox_patch().set_edgecolor('black')

plt.tight_layout()
plt.show()

# %%
i = 0
key_root = {}
for k in root.keys():
    if root[k]['class'] == '1':
        key_root[k] = root[k]
        key_root[k]['X_population'] = train_X.loc[train_y.index[train_y > 70], :][tree_clf.apply(
            selected_train_X.loc[train_y.index[train_y > 70], :].values
            ) == int(k)].to_json(orient="split")

# pd.read_json(key_root[k]['X_population'], orient='split')

# %%
import json
with open("key_root.json","w") as f:
    json.dump(key_root, f)
    
# %%
print('time cost : {t:.4f}'.format(t = time()-start))

# %%







# %%
