import numpy as np

def compute_q(p):
    if p != np.Inf and p > 1:
        q = p / (p - 1)
    elif p == 1:
        q = np.Inf
    else:
        q = 1
    return q

def compute_adv_attack(error, jac, ord=2.0):
    """Compute one step of the adversarial attack with unitary p-norm.

    :param error:
        A numpy array of shape = (n_points,) containing (y_pred - y_true)
    :param jac:
        A numpy array of shape = (n_points, input_dim) giving the Jacobian matrix.
        I.e. the derivative of the error in relation to the parameters. For linear model
        the Jacobian should be the same for all points. In this case, just use
        shape = (1, input_dim) for the same result with less computation.
    :param ord:
        The p-norm is bounded in the adversarial attack. `ord` gives which p-norm is used
        ord = 2 is the euclidean norm. `ord` can a float value greater then or equal to 1 or np.inf,
        (for the infinity norm).
    :return:
        An array containing `delta_x` of shape = (n_points, n_parameters)
        which should perturbate the input. The p-norm of each row is equal to 1.
        In order to obtain the adversarial attack bounded by `e` just multiply it
        `delta_x`.
    """
    p = ord
    if p < 1:
        raise ValueError('`ord` is float value. 1<=ord<=np.inf.'
                         'ord = {} is not valid'.format(p))

    # Given p compute q
    if p == np.inf:
        magnitude = np.ones_like(jac)
    elif p == 1:
        magnitude = np.array(np.max(jac, axis=-1, keepdims=True) == jac, dtype=np.float)
    else:
        # Compute magnitude (this follows from the case the holder inequality hold:
        # i.e. see Ash p. 96 section 2.4 exercise 4)
        q = p / (p - 1)
        magnitude = np.abs(jac) ** (q / p)
    dx = np.sign(jac) * magnitude
    # rescale
    norm = np.linalg.norm(dx, ord=p, axis=-1, keepdims=True)
    dx = dx / norm if norm > 0 else dx
    # Compute delta_x
    delta_x = dx * np.sign(error)[:, None]

    return delta_x


def compute_pgd_attack(x, y, mdl,  max_perturb, step_size=None, ord=2.0, steps=100, verbose=False):
    """Projected gradient descent for arbitrary p-norm.

    :param x:
        A numpy array of shape = (n_points, input_dim) containing the model input.
    :param y:
        A numpy array of shape = (n_points,) containing the target.
    :param fn:
        Function with signature `mdl(X)-> y_pred, jac`. Where `y_pred` has shape = (n_points,) and and
        `jac` has shape (n_points, input_dim).
    :param max_perturb:
        The maximum lp norm allowed for the perturbation
    :param step_size:
        Step size of the iteration. When None just
    :param ord:
        Order of the norm considered. `ord` can a float value greater then or equal to 1 or np.inf,
        (for the infinity norm).
    :param steps:
        Number of iterations used in the procedure
    :param verbose:
        When true print iterations.
    :return:
        Returns the perturbation delta_x which should be applied to the
        input in order to generate the desired outcome.
    """
    if step_size is None:
        step_size = 1.4 * max_perturb / steps
    # initialize perturbation with zeros
    n_points, input_dim = x.shape
    delta_x = np.zeros((n_points, input_dim))

    # Iterations
    for i in range(steps):
        # Compute jacobian and error
        y_pred, jac = mdl(x+delta_x)
        error = y_pred - y
        # First order update
        delta_x += step_size * compute_adv_attack(error, jac, ord=ord)
        # projection step
        delta_x_norm = np.linalg.norm(delta_x, ord=ord, axis=-1, keepdims=True)
        delta_x = np.where(delta_x_norm <= max_perturb, delta_x, delta_x/delta_x_norm * max_perturb)
        if verbose:
            print('{} | {} | {} | {}'.format(np.mean(error**2), np.min(delta_x_norm), np.median(delta_x_norm), np.max(delta_x_norm)))
    return delta_x





