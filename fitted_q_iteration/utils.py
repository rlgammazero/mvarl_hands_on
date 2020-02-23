import numpy as np 
import gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_grid(x, y):
    m = np.meshgrid(x, y, copy=False, indexing='ij')
    return np.vstack(m).reshape(2, -1).T

def plot_solution(env, feature_map, states, actions, theta_hat, gamma):
    SA = make_grid(states, actions)
    S, A = SA[:, 0], SA[:, 1]

    K, cov = env.computeOptimalK(gamma), 0.001
    print('Optimal K: {} Covariance S: {}'.format(K, cov))

    # Optimal Q function
    Q_fun_ = np.vectorize(lambda s, a: env.computeQFunction(s, a, K, cov, gamma, 1))
    Q_fun = lambda X: Q_fun_(X[:, 0], X[:, 1])
    Q_opt = Q_fun(SA)

    # Fitted Q function
    Q_fitted = np.zeros((len(states), len(actions)))
    for ii, ss in enumerate(states):
        for jj, aa in enumerate(actions):
            Q_fitted[ii, jj] = theta_hat.dot(feature_map.map(ss, aa))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(S, A, Q_opt, label='optimal Q function')
    ax.scatter(S, A, Q_fitted, label='fitted Q function')
    plt.legend()
    plt.show()