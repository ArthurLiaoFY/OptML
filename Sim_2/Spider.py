# %%
import numpy as np

class Spider():
    def __init__(self, 
                 cost_func, 
                 udf_func, 
                 n_dim, 
                 population_value,
                 udf_UPH,
                 changeable_group, # np.array([[AOI group idx], [ICT group idx], [UV group idx], ...])
                 size_pop=50, 
                 max_iter=200, 
                 prob_mut=0.3,
                 ):
        self.udf_func = udf_func
        self.cost_func = cost_func
        self.n_dim = n_dim
        self.population_value = population_value # pd.read_json(popuation_value, orient='split').values
        self.udf_UPH = udf_UPH
        self.changeable_group = changeable_group
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.prob_mut = prob_mut
        self.init_weight = 1
        self.population_weight = [self.init_weight for i in range(len(self.population_value))]

        self.history_best_X = []
        self.history_min_cost = []

    def mutation(self):
        self.X_init_idx, X_end_idx = np.random.choice(
            np.arange(self.population_value.shape[0]), 
            size=2, 
            replace=False, 
            p=self.population_weight/np.sum(self.population_weight))
        self.X = self.population_value[self.X_init_idx]
        self.V = self.X + np.random.uniform(0, 1)*(self.population_value[X_end_idx] - self.X)

    def crossover(self):
        self.U = self.V
        for group in self.changeable_group:
            if len(group)==0:
                continue
            else:
                if np.random.uniform(0, 1) < self.prob_mut:
                    to_idx = np.random.choice(
                        a=group, 
                        size=np.random.choice(np.append(0, np.arange(2, len(group))), 1)[0], 
                        replace=False)
                    
                    from_idx = np.random.choice(
                        a=to_idx, 
                        size=len(to_idx), 
                        replace=False)
                    
                    for f, t in zip(from_idx, to_idx):
                        self.U[t] = self.V[f]
                else:
                    continue

    def selection(self):
        cost_values = np.array([
                self.x2y(self.X), 
                self.x2y(self.V), 
                self.x2y(self.U)])
        # print(self.udf_func(self.V[np.newaxis, :]) >= self.udf_UPH)
        select_mask = [True, 
                       (self.udf_func(self.V[np.newaxis, :]) >= self.udf_UPH)[0], 
                       (self.udf_func(self.U[np.newaxis, :]) >= self.udf_UPH)[0]]
        

        min_cost = np.min(cost_values[select_mask])
        argmin_cost = np.argmin(cost_values[select_mask])

        if np.sum(select_mask) == 1:
            pass
        
        elif np.sum(select_mask) == 3:
            self.X = self.V if argmin_cost == 1 else self.U if argmin_cost == 2 else self.X
        else:
            if argmin_cost == 0:
                print('X is replaced by X')
                pass
            elif argmin_cost == 1:
                if self.udf_func(self.V) < self.udf_UPH:
                    print('X is replaced by U')
                    self.X = self.U
                else:
                    print('X is replaced by V')
                    self.X = self.V

        self.population_value = np.insert(
            self.population_value, 
            self.X_init_idx, 
            self.X, 
            axis=0)
        
        self.population_weight = np.insert(
            self.population_weight, 
            self.X_init_idx, 
            self.init_weight+1, 
            axis=0)

        return min_cost

    def x2y(self, x):
        return self.cost_func(x)

    def run(self, max_iter=None):
        if max_iter:
            self.max_iter = max_iter
        for i in range(self.max_iter):
            pop_cost = []
            pop_x = []
            for pop in range(self.size_pop):
                self.mutation()
                self.crossover()
                pop_cost.append(self.selection())
                pop_x.append(self.X)
                
            self.init_weight += 1
            self.history_min_cost.append(np.min(pop_cost))
            self.history_best_X.append(list(pop_x[np.argmin(pop_cost)]))
        
        self.best_y = np.min(self.history_min_cost)
        self.best_x = self.history_best_X[np.argmin(self.history_min_cost)]

        return self.best_x, self.best_y, self.udf_func(np.array(self.best_x)[np.newaxis, :])[0]
    
# %%
# if __name__ == "__main__":
config={}
config['seed']=1122
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation, PillowWriter
mpl.rcParams["contour.linewidth"] = 0.7

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.data[:, 2]
xgb = XGBRegressor(random_state=config['seed'])
xgb.fit(X, y)
tree_ = DecisionTreeClassifier(
    min_samples_leaf=50, 
    random_state=config['seed'])
tree_.fit(X, (xgb.predict(X) >= 4).astype(int).astype(str))

spider = Spider(cost_func=lambda x:(x[0] - X.mean(axis=0)[0])**2 + (x[1] - X.mean(axis=0)[1])**2, 
                udf_func=xgb.predict, 
                n_dim=2, 
                population_value=X[(tree_.apply(X) == 2) & (y>=4)],
                udf_UPH=4 ,
                changeable_group=[[]], # np.array([[AOI group idx], [ICT group idx], [UV group idx], ...])
                size_pop=50, 
                max_iter=200, 
                prob_mut=0.3)

spider.run()


plt.plot(spider.history_min_cost)
plt.title('convergence trend')
plt.show()

fig = plt.figure(figsize=(14, 5))
ax0 = fig.add_subplot(131, projection="3d")

f  = lambda x, y: (x-X.mean(axis=0)[0])**2 + (y-X.mean(axis=0)[1])**2
ax0xmin, ax0xmax, ax0xstep = 5.4, 8.2, 0.01
ax0ymin, ax0ymax, ax0ystep = 2, 4, 0.01
X_, Y_ = np.meshgrid(np.arange(ax0xmin, ax0xmax + ax0xstep, ax0xstep), np.arange(ax0ymin, ax0ymax + ax0ystep, ax0ystep))
Z_ = f(X_, Y_)

ax0.plot_surface(X_, Y_, Z_, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
ax0.contour(X_, Y_, Z_, 15, cmap="autumn_r", linestyles="solid", offset=-1)
# ax0.contour(X_, Y_, Z_, 15, colors="k", linestyles="solid")
ax0.plot(*np.array([X.mean(axis=0)[0],  X.mean(axis=0)[1], 0]).reshape(-1, 1), 'r*', markersize=18)
ax0.view_init(elev=25, azim=125, roll=0)

ax0.set_xlabel('$x$')
ax0.set_ylabel('$y$')

ax0.set_xlim((ax0xmin, ax0xmax))
ax0.set_ylim((ax0ymin, ax0ymax))

ax1 = fig.add_subplot(132, projection="3d")

mean_distance = np.array([
    np.abs(np.array(spider.history_best_X).max(axis=0) - X.mean(axis=0)), 
    np.abs(np.array(spider.history_best_X).min(axis=0) - X.mean(axis=0))]
    ).max(axis=0)

ax1xmin, ax1xmax, ax1xstep = X.mean(axis=0)[0]-mean_distance[0]*1.2, X.mean(axis=0)[0]+mean_distance[0]*1.2 , 0.0005
ax1ymin, ax1ymax, ax1ystep = X.mean(axis=0)[1]-mean_distance[1]*1.2, X.mean(axis=0)[1]+mean_distance[1]*1.2 , 0.0005

X_, Y_ = np.meshgrid(np.arange(ax1xmin, ax1xmax + ax1xstep, ax1xstep), np.arange(ax1ymin, ax1ymax + ax1ystep, ax1ystep))
Z_ = f(X_, Y_)

ax1.plot_surface(X_, Y_, Z_, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)


ax1.plot(*np.array([X.mean(axis=0)[0],  X.mean(axis=0)[1], 0]).reshape(-1, 1), 'r*', markersize=18)
ax1.view_init(elev=25, azim=125, roll=0)

ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')

ax1.set_xlim((ax1xmin, ax1xmax))
ax1.set_ylim((ax1ymin, ax1ymax))

ax2 = fig.add_subplot(133)
ax2.set_xlim((5.4, 8.2))
ax2.set_ylim((2, 4))
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
def animation_func(i):
    ax2.cla()
    ax2.set_xlim((5.4, 8.2))
    ax2.set_ylim((2, 4))
    ax2.hlines(y=X.mean(axis=0)[1], xmin=5.4, xmax=8.2, colors='black', linestyles='dashed')
    ax2.vlines(x=X.mean(axis=0)[0], ymin=2, ymax=4, colors='black', linestyles='dashed')
    ax2.scatter(
        x=spider.population_value[spider.population_weight < i][:,0],
        y=spider.population_value[spider.population_weight < i][:,1],
        c = 'gray', 
        alpha = 0.1)
    ax2.scatter(
        x=spider.population_value[spider.population_weight == i][:,0],
        y=spider.population_value[spider.population_weight == i][:,1],
        c='blue'
        )

plt.show()
# %%

def descent_animation_func(i):
    ax0.plot3D(
        [spider.history_best_X[i][0], spider.history_best_X[i+1][0]], 
        [spider.history_best_X[i][1], spider.history_best_X[i+1][1]], 
        [spider.cost_func(spider.history_best_X[i]), 
        spider.cost_func(spider.history_best_X[i+1])], color='gray')
    ax0.scatter(
        [spider.history_best_X[i][0], spider.history_best_X[i+1][0]], 
        [spider.history_best_X[i][1], spider.history_best_X[i+1][1]], 
        [spider.cost_func(spider.history_best_X[i]),
        spider.cost_func(spider.history_best_X[i+1])], color='b')
    ax0.set_title('Iteration {i}'.format(i=i+1))


    ax1.plot3D(
        [spider.history_best_X[i][0], spider.history_best_X[i+1][0]], 
        [spider.history_best_X[i][1], spider.history_best_X[i+1][1]], 
        [spider.cost_func(spider.history_best_X[i]), 
        spider.cost_func(spider.history_best_X[i+1])], color='gray')
    ax1.scatter(
        [spider.history_best_X[i][0], spider.history_best_X[i+1][0]], 
        [spider.history_best_X[i][1], spider.history_best_X[i+1][1]], 
        [spider.cost_func(spider.history_best_X[i]),
        spider.cost_func(spider.history_best_X[i+1])], color='b')

    ax1.set_title('Iteration {i}'.format(i=i+1))

    ax2.cla()
    ax2.set_xlim((5.4, 8.2))
    ax2.set_ylim((2, 4))
    ax2.hlines(y=X.mean(axis=0)[1], xmin=5.4, xmax=8.2, colors='black', linestyles='dashed')
    ax2.vlines(x=X.mean(axis=0)[0], ymin=2, ymax=4, colors='black', linestyles='dashed')
    ax2.scatter(
        x=spider.population_value[spider.population_weight < i][:,0],
        y=spider.population_value[spider.population_weight < i][:,1],
        c = 'gray', 
        alpha = 0.1)
    ax2.scatter(
        x=spider.population_value[spider.population_weight == i][:,0],
        y=spider.population_value[spider.population_weight == i][:,1],
        c='blue'
        )
    ax2.set_title('Iteration {i}'.format(i=i+1))



animation = FuncAnimation(
    fig=fig, 
    func=descent_animation_func, 
    frames=spider.max_iter-1,
    interval=10, 
    repeat_delay=50, 
    repeat=True, 
    )
animation.save(
    filename='iris_descent_example.gif', dpi=300, writer=PillowWriter(fps=25))



# %%
