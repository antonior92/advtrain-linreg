# Import packages.
import cvxpy as cp
import numpy as np
import numpy.linalg as linalg
import scipy.linalg as linalg

from advtrain import AdversarialTraining, compute_q


class Ridge:
    def __init__(self, X, y):
        self.u, self.s, self.vh = linalg.svd(X, full_matrices=False, compute_uv=True)
        self.y = y

    def __call__(self, regularization):
        u, s, vh = self.u, self.s, self.vh
        y = self.y
        prod_aux = s / (regularization + s ** 2)  # If S = diag(s) => P = inv(S.T S + ridge * I) S.T => prod_aux = diag(P)
        estim_param = (prod_aux * (y @ u)) @ vh  # here estim_param = V P U.T
        return estim_param


class MinimumNorm():
    def __init__(self, X, y, p, **kwargs):
        ntrain, nfeatures = X.shape

        param = cp.Variable(nfeatures)
        objective = cp.Minimize(cp.pnorm(param, p=p))
        constraints = [y == X @ param, ]
        prob = cp.Problem(objective, constraints)

        try:
            result = prob.solve(**kwargs)
            self.param = param.value
            self.alpha = constraints[0].dual_value
        except:
            self.param = np.zeros(nfeatures)
            self.alpha = np.zeros(ntrain)
        self.prob = prob
        self.ntrain = ntrain

    def __call__(self):
        return self.param

    def adv_radius(self):
        return 1 / (self.ntrain * np.max(np.abs(self.alpha)))


def adversarial_training(X, y, p, eps, **kwargs):
    """Compute parameter for linear model trained adversarially with unitary p-norm.

    :param X:
        A numpy array of shape = (n_points, input_dim) containing the inputs
    :param y:
        A numpy array of shape = (n_points,) containing true outcomes
    :param p:
        The p-norm the adversarial attack is bounded. `p` gives which p-norm is used
        p = 2 is the euclidean norm. `p` can a float value greater then or equal to 1 or np.inf,
        (for the infinity norm).
    :param eps:
        The magnitude of the attack during the training
    :return:
        An array containing the adversarially estimated parameter.
    """
    advtrain = AdversarialTraining(X, y, p)
    return advtrain(eps, **kwargs)


def ridge(X, y, regul, **kwargs):
    advtrain = Ridge(X, y,)
    return advtrain(regul, **kwargs)


def adversarial_training_randproj(X, S,  y, p, eps, **kwargs):
    """Compute parameter for linear model trained adversarially with unitary p-norm.

    :param X:
        A numpy array of shape = (n_points, inp_dim) containing the inputs
    :param S:
        A numpy array of shape = (n_features, inp_dim) containing the random projection matrix
    :param y:
        A numpy array of shape = (n_points,) containing true outcomes
    :param p:
        The p-norm the adversarial attack is bounded. `p` gives which p-norm is used
        p = 2 is the euclidean norm. `p` can a float value greater then or equal to 1 or np.inf,
        (for the infinity norm).
    :param eps:
        The magnitude of the attack during the trainign
    :return:
        An array containing the adversarially estimated parameter.
    """
    n_points, inp_dim = X.shape
    n_features, inp_dim = S.shape

    q = compute_q(p)

    # Formulate problem
    param = cp.Variable(n_features)
    param_d = S.T @ param
    param_norm = cp.pnorm(param_d,  p=q)
    abs_error = cp.abs(X @ param_d - y)
    adv_loss = 1 / n_points * cp.sum((abs_error + eps * param_norm)**2)

    prob = cp.Problem(cp.Minimize(adv_loss))
    try:
        prob.solve(**kwargs)
        param0 = param.value
    except:
        param0 = np.zeros(n_features)

    return param0


def lasso_cvx(X, y, eps, **kwargs):
    """Compute parameter for linear model using lasso (using cvxpy).

    :param X:
        A numpy array of shape = (n_points, input_dim) containing the inputs
    :param y:
        A numpy array of shape = (n_points,) containing true outcomes
    :param eps:
        The magnitude of the attack during the trainign
    :return:
        An array containing the adversarially estimated parameter.
    """
    m, n = X.shape

    # Formulate problem
    param = cp.Variable(n)
    param_norm = cp.pnorm(param,  p=1)
    square_error = cp.sum((X @ param - y)**2)
    adv_loss = 1 / (2 * m) * square_error + eps * param_norm

    prob = cp.Problem(cp.Minimize(adv_loss))
    try:
        prob.solve(**kwargs)
        param0 = param.value
    except:
        param0 = np.zeros(n)
    return param0


def sqrt_lasso(X, y, eps, **kwargs):
    """Compute parameter for linear model using square root lasso (using cvxpy).

    :param X:
        A numpy array of shape = (n_points, input_dim) containing the inputs
    :param y:
        A numpy array of shape = (n_points,) containing true outcomes
    :param eps:
        The magnitude of the attack during the training
    :return:
        An array containing the adversarially estimated parameter.
    """
    m, n = X.shape

    # Formulate problem
    param = cp.Variable(n)
    param_norm = cp.pnorm(param,  p=1)
    error_norm = cp.pnorm((1 / m) * (X @ param - y), p=2)
    loss = error_norm + eps * param_norm

    prob = cp.Problem(cp.Minimize(loss))
    try:
        prob.solve(**kwargs)
        param0 = param.value
    except:
        param0 = np.zeros(n)
    return param0


def minl1norm_solution(X, y, **kwargs):
    m, n = X.shape

    param = cp.Variable(n)
    objective = cp.Minimize(cp.pnorm(param, p=1))
    constraints = [y == X @ param, ]
    prob = cp.Problem(objective, constraints)

    result = prob.solve(**kwargs)
    return param.value


def get_max_alpha(X, y, p=2):
    n, m = X.shape
    q = compute_q(p)

    var = cp.Variable(n)

    obj = cp.Maximize(var @ y)
    constr = [cp.pnorm(X.T @ var, p=p) <= 1,]
    prob = cp.Problem(obj, constr)
    result = prob.solve()
    return 1 / (n * np.max(np.abs(var.value)))


# Define and solve the CVXPY problem.
if __name__ == '__main__':
    # Generate data.
    m = 20
    n = 23
    np.random.seed(1)
    X = np.random.randn(m, n)
    y = np.random.randn(m)

    param = adversarial_training(X, y, 2, 0.1)
    param_lasso = lasso_cvx(X, y, 0.1)
    param_ridge = ridge(X, y, 0.1)
    param_minl1norm = minl1norm_solution(X, y)
    param_sqrt_lasso = sqrt_lasso(X, y, 0.1)

    print(np.linalg.norm(X @ param - y))
    print(np.linalg.norm(param))

    print(np.linalg.norm(X @ param_lasso - y))
    print(np.linalg.norm(param_lasso))

    print(np.linalg.norm(X @ param_ridge - y))
    print(np.linalg.norm(param_ridge))

    print(np.linalg.norm(X @ param_minl1norm - y))
    print(np.linalg.norm(param_minl1norm))

    print(np.linalg.norm(X @ param_sqrt_lasso - y))
    print(np.linalg.norm(param_sqrt_lasso))
