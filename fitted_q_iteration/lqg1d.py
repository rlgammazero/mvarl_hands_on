"""classic Linear Quadratic Gaussian Regulator task"""
from numbers import Number
import numpy as np
from gym import spaces

"""
Linear quadratic gaussian regulator task.

References
----------
  - Jan  Peters  and  Stefan  Schaal,
    Reinforcement  learning of motor  skills  with  policy  gradients,
    Neural  Networks, vol. 21, no. 4, pp. 682-697, 2008.

"""


class LQG1D(object):

    def __init__(self, initial_state_type='random'):
        assert initial_state_type in ['random', 'fixed']
        self.max_pos = 20
        self.max_action = 40
        self.sigma_noise = -1
        self.A = np.array([1]).reshape((1, 1))
        self.B = np.array([1]).reshape((1, 1))
        self.Q = np.array([0.5]).reshape((1, 1))
        self.R = np.array([0.5]).reshape((1, 1))

        self.initial_states = [-10, -5, 5, 10]
        self.initial_states_type = initial_state_type

        # initialize state
        self.reset()

        # observation and action spaces
        self.state_low = np.array([-10.0])
        self.state_high = np.array([10.0])

        self.action_low = np.array([-8.0])
        self.action_high = np.array([8.0])

        self.observation_space = spaces.Box(self.state_low, self.state_high, dtype=np.float64)
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float64)


    def step(self, action, render=False):
        u = np.clip(action, -self.max_action, self.max_action)
        noise = 0
        if self.sigma_noise > 0:
            noise = np.random.randn() * self.sigma_noise
        xn = np.clip(np.dot(self.A, self.state) + np.dot(self.B, u) + noise, -self.max_pos, self.max_pos)
        cost = np.dot(self.state,
                      np.dot(self.Q, self.state)) + \
            np.dot(u, np.dot(self.R, u))

        self.state = np.array(xn.ravel())
        return self.get_state(), -np.asscalar(cost), False, {}

    def reset(self, state=None):
        if state is None:
            if self.initial_states_type == 'random':
                self.state = np.array([np.random.uniform(low=-10,
                                                          high=10)])
            else:
                idx = np.random.randint(low=0, high=len(self.initial_states))
                self.state = np.array([self.initial_states[idx]])
        else:
            self.state = np.array(state)

        return self.get_state()

    def get_state(self):
        return np.array(self.state)

    def _computeP2(self, K, gamma):
        """
        This function computes the Riccati equation associated to the LQG
        problem.
        Args:
            K (matrix): the matrix associated to the linear controller K * x

        Returns:
            P (matrix): the Riccati Matrix

        """
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if np.array_equal(self.A, I) and np.array_equal(self.B, I):
            P = (self.Q + np.dot(K.T, np.dot(self.R, K))) / (I - gamma *
                                                             (I + 2 * K + K **
                                                              2))
        else:
            tolerance = 0.0001
            converged = False
            P = np.eye(self.Q.shape[0], self.Q.shape[1])
            while not converged:
                Pnew = self.Q + gamma * np.dot(self.A.T,
                                                    np.dot(P, self.A)) + \
                       gamma * np.dot(K.T, np.dot(self.B.T,
                                                       np.dot(P, self.A))) + \
                       gamma * np.dot(self.A.T,
                                           np.dot(P, np.dot(self.B, K))) + \
                       gamma * np.dot(K.T,
                                           np.dot(self.B.T,
                                                  np.dot(P, np.dot(self.B,
                                                                   K)))) + \
                       np.dot(K.T, np.dot(self.R, K))
                converged = np.max(np.abs(P - Pnew)) < tolerance
                P = Pnew
        return P

    def computeOptimalK(self, gamma):
        """
        This function computes the optimal linear controller associated to the
        LQG problem (u = K * x).

        Returns:
            K (matrix): the optimal controller

        """
        P = np.eye(self.Q.shape[0], self.Q.shape[1])
        for i in range(100):
            K = -gamma * np.dot(np.linalg.inv(
                self.R + gamma * (np.dot(self.B.T, np.dot(P, self.B)))),
                                       np.dot(self.B.T, np.dot(P, self.A)))
            P = self._computeP2(K, gamma)
        K = -gamma * np.dot(np.linalg.inv(self.R + gamma *
                                               (np.dot(self.B.T,
                                                       np.dot(P, self.B)))),
                                 np.dot(self.B.T, np.dot(P, self.A)))
        return K

    def computeQFunction(self, x, u, K, Sigma, gamma, n_random_xn=100):
        """
        This function computes the Q-value of a pair (x,u) given the linear
        controller Kx + epsilon where epsilon \sim N(0, Sigma).
        Args:
            x (int, array): the state
            u (int, array): the action
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
            the controller action
            n_random_xn: the number of samples to draw in order to average over
            the next state

        Returns:
            Qfun (float): The Q-value in the given pair (x,u) under the given
            controller

        """
        if isinstance(x, Number):
            x = np.array([x])
        if isinstance(u, Number):
            u = np.array([u])
        if isinstance(K, Number):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, Number):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self._computeP2(K, gamma=gamma)
        Qfun = 0
        for i in range(n_random_xn):
            noise = 0
            if self.sigma_noise > 0:
                noise = np.random.randn() * self.sigma_noise
            action_noise = np.random.multivariate_normal(
                np.zeros(Sigma.shape[0]), Sigma, 1)
            nextstate = np.dot(self.A, x) + np.dot(self.B,
                                                   u + action_noise) + noise
            Qfun -= np.dot(x.T, np.dot(self.Q, x)) + \
                np.dot(u.T, np.dot(self.R, u)) + \
                gamma * np.dot(nextstate.T, np.dot(P, nextstate)) + \
                (gamma / (1 - gamma)) * \
                np.trace(np.dot(Sigma,
                                self.R + gamma *
                                np.dot(self.B.T, np.dot(P, self.B))))
        Qfun = np.asscalar(Qfun) / n_random_xn
        return Qfun