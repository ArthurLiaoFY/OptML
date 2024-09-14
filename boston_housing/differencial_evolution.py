# %%
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from sko.modified_DE import DE
from sko.tools import set_run_mode
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
mpl.rcParams['font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus']=False 
config = {}
config['seed'] = 1122
with open('key_root.json') as f:
    key_root = json.load(f)

df = pd.read_csv('uph_opt.csv', encoding='gbk').dropna()
train_X, val_X, train_y, val_y = train_test_split(
#     df.drop(columns=['Begin_Hour', 'LINE_NAME', 'MODEL_NAME', 'Date', 'UPH', 'UPH_Equv', 'MO_NUMBER', 'REAL_QTY_Equv']), 
    df.drop(columns=['UPH', 'UPH_Equv']), 
    df['UPH'], 
    train_size=0.8,
    random_state=config['seed'])

xgb = pickle.load(open('xgb_response_surface.pkl', "rb"))
# %%
def obj_func(p):
    '''
    sum ( w_i * ( x_i - x_mean) ** 2 )
    '''
    return float(sum(config['weight_list'] * (np.array(p) - config['sample_mean'])**2))

config['weight_list'] = [1 for i in range(train_X.shape[1])]
config['sample_mean'] = train_X.mean().values
set_run_mode(obj_func, 'multithreading')
# %% drop some columns
for k, v in key_root.items():
    print(key_root[k]['constraint_order'])

# %%
'''
min f(x1, x2) = f(x1, x2)
s.t.
    x2 >= 0.5  => 0.5-x2 <= 0
    x2 - 4*x1 >= 1  => 1-x2+4*x1 <= 0
    -1.75 <= x1, x2 <= 1.75
'''
# constraint_ueq = []
machine_series = train_X.columns.str.rsplit(pat='-', n=1, expand=True)\
    .to_frame(index=False).dropna()[0].str.replace('-', '')

constraint_ueq = []
column_map = pd.Series(train_X.columns)

for machine in machine_series.unique():
    selected_cols = set(
        train_X.columns[machine_series[machine_series.isin([machine])].index]\
            .to_list())
    print(selected_cols)
    # constraint_eq.append(eval('lambda x:'\
    #                     + '+'.join('x[{idx}]'.format(idx=idx) for idx in column_map.index[column_map.isin(selected_cols)])\
    #                         +'-3600'))

    constraint_ueq.append('lambda x:'\
                        + '+'.join('x[{idx}]'.format(idx=idx) for idx in column_map.index[column_map.isin(selected_cols)])\
                            +'-3600')

# %%
def Optimize(constraint_order, constraint_ueq, patience=50):
    counter = 0
    min_y = np.Inf
    constraint_ueq = []
    for c in constraint_order:
        constraint_ueq.append(eval(c))


    max_iter = 1000
    size_pop = 100
    # DE
    de = DE(
        func=obj_func, 
        n_dim=train_X.shape[1], 
        size_pop=size_pop, 
        max_iter=max_iter, 
        lb=[0 for i in range(train_X.shape[1])], 
        ub=[3600 for i in range(train_X.shape[1])],
        # constraint_eq=constraint_eq,
        constraint_ueq=constraint_ueq,

        )
    # while var_decay
    while counter < patience:
        best_x, best_obj_func = de.run(5)
        if round(np.min(de.all_history_Y[-100:]), 4) >= round(min_y, 4):
            counter += 1
            print('{curr_min:.4f} >= {hist_min:.4f}, Counter : {count}/{total}'.format(
                curr_min=np.min(de.all_history_Y[-100:]),
                hist_min=min_y,
                count=counter,
                total=patience
            ))

        else:
            print('Min value refresh, {curr_min:.4f} < {hist_min:.4f}'.format(
                curr_min=np.min(de.all_history_Y[-100:]),
                hist_min=min_y
            ))
            if counter > 0:
                print('Counter Reset')
                counter = 0

            min_y = np.min(de.all_history_Y[-100:])

    # DE plot
    Y_history = pd.DataFrame(de.all_history_Y)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.title('收斂趨勢')
    plt.show()
    return best_x, best_obj_func, de.generation_best_X
# %%
for k, v in key_root.items():
    key_root[k]['X_opt'], key_root[k]['Cost'], key_root[k]['Path'] = Optimize(v['constraint_order'], constraint_ueq)
    key_root[k]['mean_diff'] = key_root[k]['X_opt']-train_X.mean(axis=0)
    key_root[k]['mean_diff_ratio'] = (key_root[k]['X_opt']-train_X.mean(axis=0)).abs()/train_X.mean(axis=0)  
    key_root[k]['predicted'] = xgb.predict((
        key_root[k]['X_opt'])\
            .reshape(1,-1))

    # save value to csv
    key_root[k]['mean_diff'].reset_index().rename(columns={'index':'machine_status', 0:'mean_difference'}).merge(
        key_root[k]['mean_diff_ratio'].reset_index().rename(columns={'index':'machine_status', 0:'mean_difference_ratio'}),
        how='inner',
        on='machine_status'
    ).merge(
        pd.DataFrame(key_root[k]['X_opt'], index=key_root[k]['mean_diff'].index).reset_index().rename(columns={'index':'machine_status', 0:'X_opt'}),
        how='inner',
        on='machine_status'
        ).merge(
            train_X.mean(axis=0).reset_index().rename(columns={'index':'machine_status', 0:'mean_value'}),
            how='inner',
            on='machine_status')\
                .reset_index(drop=True)[['machine_status','X_opt', 'mean_value','mean_difference','mean_difference_ratio']].to_csv('X_info_'+str(k)+'.csv', encoding='gbk')



# %%
