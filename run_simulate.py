# %%
from configparser import ConfigParser

from common.estimate_surface import EstimateSurface
from common.metrics_fn import R_square, rmse
from common.optimize_response import optimize_f_hat
from common.plot_fn import plot_obj_surface
from simulate.simulate_data import SimulateData

config = ConfigParser()
config.read("config.ini")

seed = int(config["simulate"].get("seed"))
simulate_size = int(config["simulate"].get("simulate_size"))
variation_ratio = float(config["simulate"].get("variation_ratio"))
train_size_ratio = float(config["simulate"].get("train_size_ratio"))
func = config["simulate"].get("func")
model_type = config["simulate"].get("model_type")
opt_type = config["simulate"].get("opt_type")
x1_step = int(config["simulate"].get("x1_step"))
x2_step = int(config["simulate"].get("x2_step"))
max_iter = int(config["simulate"].get("max_iter"))
size_pop = int(config["simulate"].get("size_pop"))

sd = SimulateData(
    seed=seed,
    simulate_size=simulate_size,
    variation_ratio=variation_ratio,
    sim_func=func,
)
X_train, X_val, y_train, y_val = sd.get_data(train_size_ratio=train_size_ratio)

sm = EstimateSurface(seed=seed, model_type=model_type)
sm.fit_model(X=X_train, y=y_train)
y_hat = sm.pred_model(valid_X=X_val)

print(rmse(y=y_val, y_hat=y_hat))
print(R_square(y=y_val, y_hat=y_hat))

opt, a, b = optimize_f_hat(
    obj_func=sm.model.predict,
    constraint_ueq=sd.constraint_ueq,
    max_iter=max_iter,
    size_pop=size_pop,
    x_max=sd.x_max,
    x_min=sd.x_min,
    opt_type=opt_type,
)
print(a)
print(b)
# %%
plot_obj_surface(
    pso_opt=opt,
    f_hat=sm.model,
    max_iter=max_iter,
    x_max=sd.x_max,
    x_min=sd.x_min,
    x1_step=x1_step,
    x2_step=x2_step,
    animate=True,
    desc=func,
)

# %%
