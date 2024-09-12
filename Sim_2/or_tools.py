# %%
import json
import pandas as pd
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
solver = pywraplp.Solver.CreateSolver("GLOP")

with open('key_root.json') as f:
    key_root = json.load(f)

train_col = ['AOI-0',
 'AOI-1',
 'AOI-2',
 'AOI-3',
 'AOI-4',
 'AOI不良分流--1',
 'AOI不良分流-0',
 'AOI不良分流-1',
 'AOI不良分流-2',
 'AOI不良分流-4',
 'AOI不良分流-7',
 'ICT不良分流-1--1',
 'ICT不良分流-1-0',
 'ICT不良分流-1-1',
 'ICT不良分流-1-4',
 'ICT不良分流-1-7',
 'ICT不良分流-2--1',
 'ICT不良分流-2-0',
 'ICT不良分流-2-1',
 'ICT不良分流-2-4',
 'ICT不良分流-2-7',
 'UV爐--1',
 'UV爐-0',
 'UV爐-1',
 'UV爐-3',
 'UV爐-4',
 '二次線體--1',
 '二次線體-0',
 '分板機-1--1',
 '分板機-1-0',
 '分板機-1-1',
 '分板機-1-2',
 '分板機-1-3',
 '分板機-1-4',
 '噴膠機(五軸)--1',
 '噴膠機(五軸)-0',
 '噴膠機(五軸)-1',
 '噴膠機(五軸)-2',
 '噴膠機(五軸)-3',
 '噴膠機(五軸)-4',
 '推板機--1',
 '推板機-0',
 '泡膠機--1',
 '泡膠機-0',
 '泡膠機-1',
 '泡膠機-2',
 '泡膠機-3',
 '泡膠機-4',
 '翻板移載機--1',
 '翻板移載機-0',
 '翻板移載機-1',
 '翻板移載機-2',
 '翻板移載機-3',
 '翻板移載機-4',
 '自動ICT-1--1',
 '自動ICT-1-0',
 '自動ICT-1-1',
 '自動ICT-1-2',
 '自動ICT-1-3',
 '自動ICT-1-4',
 '自動ICT-2--1',
 '自動ICT-2-0',
 '自動ICT-2-1',
 '自動ICT-2-2',
 '自動ICT-2-3',
 '自動ICT-2-4',
 '自動Link--1',
 '自動Link-0',
 '自動Link-1',
 '自動Link-3',
 '自動Link-4',
 '追蹤點膠機--1',
 '追蹤點膠機-0',
 '追蹤點膠機-1',
 '追蹤點膠機-2',
 '追蹤點膠機-4',
 '靜態不良分流--1',
 '靜態不良分流-0',
 '靜態不良分流-1',
 '靜態測試--1',
 '靜態測試-0',
 '靜態測試-1',
 '靜態測試-2',
 '靜態測試-3',
 '靜態測試-4',
 ' 組裝機（側蓋） -TBS-61',
 ' 組裝機（側蓋） -TBS-62',
 ' 組裝機（側蓋） -TBS-63',
 ' 組裝機（側蓋） -TBS-750',
 'AOI-TBS-5',
 'AOI-TBS-6',
 'AOI-TBS-7',
 'AOI-TBS-8',
 'CTRL鎖付一體機-TBS-2',
 'CTRL鎖付一體機-TBS-205',
 'CTRL鎖付一體機-TBS-61',
 'CTRL鎖付一體機-TBS-66',
 'CTRL鎖付一體機-TBS-780',
 'CTRL鎖付一體機-TBS-820',
 'CTRL鎖付一體機-TBS-821',
 'CTRL鎖付一體機-TBS-822',
 'CTRL鎖付一體機-TBS-823',
 'CTRL鎖付一體機-TBS-824',
 'HSK加工一體機-TBS-1',
 'HSK加工一體機-TBS-10',
 'HSK加工一體機-TBS-12',
 'HSK加工一體機-TBS-14',
 'HSK加工一體機-TBS-15',
 'HSK加工一體機-TBS-2',
 'HSK加工一體機-TBS-3',
 'HSK加工一體機-TBS-7',
 'HSK加工一體機-TBS-8',
 'HSK加工一體機-TBS-9',
 'PWR BD 鎖附一體化_1-TBS-15',
 'PWR BD 鎖附一體化_1-TBS-31',
 'PWR BD 鎖附一體化_1-TBS-32',
 'PWR BD 鎖附一體化_1-TBS-36',
 'PWR BD 鎖附一體化_1-TBS-55',
 'PWR BD 鎖附一體化_1-TBS-56',
 'PWR BD 鎖附一體化_1-TBS-57',
 'PWR BD 鎖附一體化_1-TBS-58',
 'PWR BD 鎖附一體化_1-TBS-64',
 'PWR BD 鎖附一體化_1-TBS-67',
 'PWR BD 鎖附一體化_1-TBS-71',
 'PWR BD 鎖附一體化_1-TBS-76',
 'PWR BD 鎖附一體化_2-TBS-15',
 'PWR BD 鎖附一體化_2-TBS-23',
 'PWR BD 鎖附一體化_2-TBS-27',
 'PWR BD 鎖附一體化_2-TBS-31',
 'PWR BD 鎖附一體化_2-TBS-32',
 'PWR BD 鎖附一體化_2-TBS-35',
 'PWR BD 鎖附一體化_2-TBS-36',
 'PWR BD 鎖附一體化_2-TBS-51',
 'PWR BD 鎖附一體化_2-TBS-55',
 'PWR BD 鎖附一體化_2-TBS-78',
 'nan-TBS-1',
 'nan-TBS-13',
 'nan-TBS-14',
 'nan-TBS-15',
 'nan-TBS-16',
 'nan-TBS-24',
 'nan-TBS-3',
 'nan-TBS-30',
 'nan-TBS-31',
 'nan-TBS-4',
 'nan-TBS-42',
 'nan-TBS-49',
 'nan-TBS-68',
 'nan-TBS-75',
 'nan-TBS-78',
 'nan-TBS-81',
 'nan-TBS-82',
 'nan-TBS-9',
 '分板機-1-TBS-42',
 '分板機-1-TBS-49',
 '分板機-1-TBS-75',
 '分板機-1-TBS-78',
 '分板機-1-TBS-81',
 '分板機-1-TBS-82',
 '噴膠固化一體機-TBS-20',
 '噴膠固化一體機-TBS-21',
 '噴膠固化一體機-TBS-22',
 '噴膠固化一體機-TBS-5',
 '噴膠機（五軸）_1-TBS-13',
 '噴膠機（五軸）_1-TBS-15',
 '噴膠機（五軸）_1-TBS-16',
 '噴膠機（五軸）_1-TBS-24',
 '噴膠機（五軸）_1-TBS-30',
 '噴膠機（五軸）_1-TBS-31',
 '噴膠機（五軸）_1-TBS-8',
 '噴膠機（五軸）_1-TBS-9',
 '噴膠機（五軸）_3-TBS-1',
 '噴膠機（五軸）_3-TBS-13',
 '噴膠機（五軸）_3-TBS-14',
 '噴膠機（五軸）_3-TBS-15',
 '噴膠機（五軸）_3-TBS-16',
 '噴膠機（五軸）_3-TBS-24',
 '噴膠機（五軸）_3-TBS-3',
 '噴膠機（五軸）_3-TBS-30',
 '噴膠機（五軸）_3-TBS-31',
 '噴膠機（五軸）_3-TBS-4',
 '噴膠機（五軸）_3-TBS-9',
 '成品自動測試櫃-T2測試機_1-TBS-1',
 '成品自動測試櫃-T2測試機_1-TBS-10',
 '成品自動測試櫃-T2測試機_1-TBS-13',
 '成品自動測試櫃-T2測試機_1-TBS-3',
 '成品自動測試櫃-T2測試機_1-TBS-7',
 '成品自動測試櫃-T2測試機_1-TBS-8',
 '成品自動測試櫃-T2測試機_2-TBS-1',
 '成品自動測試櫃-T2測試機_2-TBS-10',
 '成品自動測試櫃-T2測試機_2-TBS-11',
 '成品自動測試櫃-T2測試機_2-TBS-3',
 '成品自動測試櫃-T2測試機_2-TBS-4',
 '成品自動測試櫃-T2測試機_2-TBS-8',
 '成品自動測試櫃-T2測試機_2-TBS-9',
 '成品自動測試櫃-T2測試機_3-TBS-11',
 '成品自動測試櫃-T2測試機_3-TBS-13',
 '成品自動測試櫃-T2測試機_3-TBS-6',
 '成品自動測試櫃-T2測試機_3-TBS-7',
 '成品自動測試櫃-線上崩應測試機_1-TBS-11',
 '成品自動測試櫃-線上崩應測試機_1-TBS-17',
 '成品自動測試櫃-線上崩應測試機_1-TBS-18',
 '成品自動測試櫃-線上崩應測試機_2-TBS-11',
 '成品自動測試櫃-線上崩應測試機_2-TBS-17',
 '成品自動測試櫃-線上崩應測試機_2-TBS-18',
 '成品自動測試櫃-線上崩應測試機_3-TBS-20',
 '成品自動測試櫃-線上崩應測試機_4-TBS-23',
 '成品自動測試櫃-線上崩應測試機_4-TBS-24',
 '成品自動測試櫃-線上崩應測試機_5 -TBS-17',
 '成品自動測試櫃-線上崩應測試機_5 -TBS-18',
 '成品自動測試櫃-線上崩應測試機_5 -TBS-23',
 '成品自動測試櫃-線上崩應測試機_5 -TBS-24',
 '成品自動測試櫃-線上崩應測試機_5 -TBS-5',
 '成品自動測試櫃-線上崩應測試機_5 -TBS-8',
 '成品自動測試櫃-線上崩應測試機_6 -TBS-16',
 '成品自動測試櫃-線上崩應測試機_6 -TBS-17',
 '成品自動測試櫃-高壓測試機  -TBS-1',
 '成品自動測試櫃-高壓測試機  -TBS-11',
 '成品自動測試櫃-高壓測試機  -TBS-13',
 '成品自動測試櫃-高壓測試機  -TBS-16',
 '成品自動測試櫃-高壓測試機  -TBS-18',
 '成品自動測試櫃-高壓測試機  -TBS-19',
 '成品自動測試櫃-高壓測試機  -TBS-2',
 '成品自動測試櫃-高壓測試機  -TBS-21',
 '成品自動測試櫃-高壓測試機  -TBS-23',
 '成品自動測試櫃-高壓測試機  -TBS-24',
 '成品自動測試櫃-高壓測試機  -TBS-4',
 '成品自動測試櫃-高壓測試機  -TBS-5',
 '成品自動測試櫃-高壓測試機  -TBS-7',
 '成品自動測試櫃-高壓測試機  -TBS-9',
 '自动ICT-1（三工位）-TBS-120',
 '自动ICT-1（三工位）-TBS-129',
 '自动ICT-1（三工位）-TBS-130',
 '自动ICT-1（三工位）-TBS-134',
 '自动ICT-1（三工位）-TBS-136',
 '自动ICT-1（三工位）-TBS-153',
 '自动ICT-1（三工位）-TBS-176',
 '自动ICT-1（三工位）-TBS-185',
 '自动ICT-1（三工位）-TBS-198',
 '自动ICT-1（三工位）-TBS-199',
 '自动ICT-1（三工位）-TBS-203',
 '自动ICT-1（三工位）-TBS-214',
 '自动ICT-1（三工位）-TBS-216',
 '自动ICT-1（三工位）-TBS-218',
 '自动ICT-1（三工位）-TBS-230',
 '自动ICT-1（三工位）-TBS-232',
 '自动ICT-1（三工位）-TBS-240',
 '自动ICT-1（三工位）-TBS-241',
 '自动ICT-1（三工位）-TBS-243',
 '自动ICT-1（三工位）-TBS-244',
 '自动ICT-1（三工位）-TBS-247',
 '自动ICT-1（三工位）-TBS-253',
 '自动ICT-1（三工位）-TBS-256',
 '自动ICT-1（三工位）-TBS-272',
 '自动ICT-1（三工位）-TBS-274',
 '自动ICT-1（三工位）-TBS-275',
 '自动ICT-1（三工位）-TBS-276',
 '自动ICT-1（三工位）-TBS-277',
 '自动ICT-1（三工位）-TBS-279',
 '自动ICT-1（三工位）-TBS-280',
 '自动ICT-1（三工位）-TBS-281',
 '自动ICT-1（三工位）-TBS-282',
 '自动ICT-1（三工位）-TBS-283',
 '自动ICT-1（三工位）-TBS-284',
 '自动ICT-1（三工位）-TBS-285',
 '自动ICT-1（三工位）-TBS-291',
 '自动ICT-1（三工位）-TBS-292',
 '自动ICT-2（三工位）-TBS-128',
 '自动ICT-2（三工位）-TBS-130',
 '自动ICT-2（三工位）-TBS-138',
 '自动ICT-2（三工位）-TBS-142',
 '自动ICT-2（三工位）-TBS-143',
 '自动ICT-2（三工位）-TBS-151',
 '自动ICT-2（三工位）-TBS-160',
 '自动ICT-2（三工位）-TBS-169',
 '自动ICT-2（三工位）-TBS-184',
 '自动ICT-2（三工位）-TBS-188',
 '自动ICT-2（三工位）-TBS-190',
 '自动ICT-2（三工位）-TBS-199',
 '自动ICT-2（三工位）-TBS-214',
 '自动ICT-2（三工位）-TBS-216',
 '自动ICT-2（三工位）-TBS-230',
 '自动ICT-2（三工位）-TBS-232',
 '自动ICT-2（三工位）-TBS-240',
 '自动ICT-2（三工位）-TBS-241',
 '自动ICT-2（三工位）-TBS-243',
 '自动ICT-2（三工位）-TBS-253',
 '自动ICT-2（三工位）-TBS-256',
 '自动ICT-2（三工位）-TBS-272',
 '自动ICT-2（三工位）-TBS-273',
 '自动ICT-2（三工位）-TBS-275',
 '自动ICT-2（三工位）-TBS-277',
 '自动ICT-2（三工位）-TBS-279',
 '自动ICT-2（三工位）-TBS-280',
 '自动ICT-2（三工位）-TBS-283',
 '自动ICT-2（三工位）-TBS-284',
 '自动ICT-2（三工位）-TBS-291',
 '自动ICT-2（三工位）-TBS-292',
 '自動噴散熱膏機-TBS-11',
 '自動噴散熱膏機-TBS-13',
 '自動噴散熱膏機-TBS-17',
 '自動噴散熱膏機-TBS-18',
 '自動噴散熱膏機-TBS-21',
 '自動噴散熱膏機-TBS-22',
 '自動噴散熱膏機-TBS-6',
 '自動噴散熱膏機-TBS-7',
 '静态测试-TBS-17',
 '静态测试-TBS-18',
 '静态测试-TBS-20',
 '静态测试-TBS-21',
 '静态测试-TBS-61',
 '静态测试-TBS-64',
 '静态测试-TBS-66',
 '静态测试-TBS-69',
 '静态测试-TBS-7',
 'PLAN_QTY_Equv',
 'WORK_TIME',
 'LINE_RATE',
 'Hour_Duration']

status = [c for c in train_col if ('TBS' not in c) and (c not in [
    'PLAN_QTY_Equv', 
    'WORK_TIME',
    'LINE_RATE',
    'Hour_Duration'])]
tbs = [c for c in train_col if ('TBS' in c) and (c not in [
    'PLAN_QTY_Equv', 
    'WORK_TIME',
    'LINE_RATE',
    'Hour_Duration'])]
# %%
status_group = {}
machine_index = pd.Series(status).str.rsplit(pat='-', n=1, expand=True)[0].str.replace('-', '')
for machine in machine_index.unique():
    status_group[machine] = list(machine_index.index[machine_index == machine])
# %%
tbs_group = {}
machine_index = pd.Series(tbs).str.rsplit(pat='-', n=1, expand=True)[0].str.replace('-', '')
for machine in machine_index.unique():
    tbs_group[machine] = list(machine_index.index[machine_index == machine])
# %%
model = cp_model.CpModel()

status_dur = [solver.IntVar(0, 3600, var) for var in status]

tbs_error = [solver.IntVar(0, 3600, var) for var in tbs]

others = [solver.IntVar(0, solver.infinity(), var) for var in [
    'PLAN_QTY_Equv', 
    'WORK_TIME',
    'LINE_RATE',
    'Hour_Duration']]

print("Number of variables =", solver.NumVariables())


# %%
constraints = []
for idx_status, (key, vals) in enumerate(status_group.items()):
    constraints.append(solver.Constraint(3600, 3600))
    for val in vals:
        constraints[idx_status].SetCoefficient(status_dur[val], 1)


for idx_tbs, (key, vals) in enumerate(tbs_group.items()):
    constraints.append(solver.Constraint(3600, 3600))
    for val in vals:
        constraints[idx_tbs+idx_status].SetCoefficient(tbs_error[val], 1)


print("Number of constraints =", solver.NumConstraints())

# %%

# Objective function: Minimize the sum of (price-normalized) foods.
objective = solver.Objective()
for food in foods:
    objective.SetCoefficient(food, 1)
objective.SetMinimization()

status = solver.Solve()

# Check that the problem has an optimal solution.
if status != solver.OPTIMAL:
    print("The problem does not have an optimal solution!")
    if status == solver.FEASIBLE:
        print("A potentially suboptimal solution was found.")
    else:
        print("The solver could not solve the problem.")
        exit(1)

# Display the amounts (in dollars) to purchase of each food.
nutrients_result = [0] * len(nutrients)
print("\nAnnual Foods:")
for i, food in enumerate(foods):
    if food.solution_value() > 0.0:
        print("{}: ${}".format(data[i][0], 365.0 * food.solution_value()))
        for j, _ in enumerate(nutrients):
            nutrients_result[j] += data[i][j + 3] * food.solution_value()
print("\nOptimal annual price: ${:.4f}".format(365.0 * objective.Value()))

print("\nNutrients per day:")
for i, nutrient in enumerate(nutrients):
    print(
        "{}: {:.2f} (min {})".format(nutrient[0], nutrients_result[i], nutrient[1])
    )

print("\nAdvanced usage:")
print("Problem solved in ", solver.wall_time(), " milliseconds")
print("Problem solved in ", solver.iterations(), " iterations")

