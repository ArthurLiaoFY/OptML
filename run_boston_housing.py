# %%
from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini")

seed = int(config["boston_housing"].get("seed"))
# simulate_size = int(config["simulate"].get("simulate_size"))
# variation_ratio = float(config["simulate"].get("variation_ratio"))
# train_size_ratio = float(config["simulate"].get("train_size_ratio"))
# func = config["simulate"].get("func")
# model_type = config["simulate"].get("model_type")
# opt_type = config["simulate"].get("opt_type")
# max_iter = int(config["simulate"].get("max_iter"))
# size_pop = int(config["simulate"].get("siz
