import lightgbm as lgb
import time
params = {
    "max_bin": 63,
    "num_leaves": 255,
    "num_iterations": 5000,
    "learning_rate": 0.1,
    "min_data_in_leaf": 1,
    "min_sum_hessian_in_leaf": 100,
    "ndcg_eval_at": [1, 3, 5, 10],
    "device": "cpu"
}
dtrain = lgb.Dataset("higgs/higgs.train")
t0 = time.time()
gbm = lgb.train(
    params,
    train_set=dtrain
)
t1 = time.time()
print("cpu version elapse time: {}".format(t1 - t0))
# 500  83.255
# 5000 386.27542519569397