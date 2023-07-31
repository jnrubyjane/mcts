import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
def optimize_bayesian(target_function, num_samples, initial_samples, batch_size):
    # 获取目标函数
    f = target_function
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义初始训练数据
    train_X = initial_samples
    train_Y = torch.tensor([f(x) for x in initial_samples]).unsqueeze(-1)
    # 定义高斯过程回归模型
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    # 循环迭代进行优化
    for i in range(num_samples // batch_size):
        # 定义采样点
        candidate_x = torch.rand(batch_size, 1).to(device)
        # 标准化输入
        candidate_x_standardized = standardize(candidate_x, train_X.mean(0), train_X.std(0))
        # 计算期望改进
        acqf = ExpectedImprovement(model, train_Y.max())
        acquisition_value = acqf(candidate_x_standardized)
        # 选择新的采样点
        new_x = optimize_acqf(
            acq_function=acqf,
            bounds=torch.tensor([[0.0], [1.0]]).to(device),
            q=batch_size,
            num_restarts=10,
            raw_samples=100,
        ).unsqueeze(-1)
        # 更新训练数据
        new_y = torch.tensor([f(x) for x in new_x]).unsqueeze(-1)
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, new_y])
        # 更新高斯过程回归模型
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
    # 输出最优结果
    best_y, best_idx = train_Y.max(0)
    best_x = train_X[best_idx]
    return best_x, best_y.item()