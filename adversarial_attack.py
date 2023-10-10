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

