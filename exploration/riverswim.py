import numpy as np
from finite_env import FiniteEnv


class RiverSwim(FiniteEnv):
    """
    Osband, Van Roy, Russo. (More) Efficient Reinforcement Learning via Posterior Sampling. NIPS

    actions = [0: left, 1: right]
    """

    def __init__(self, n_states=6, gamma=1):
        na = 2

        self.r_max = 1.

        self.P = np.zeros((n_states, na, n_states))
        self.R = np.zeros((n_states, na))

        state_actions = []
        for s in range(n_states):
            state_actions.append([0, 1])

            if s == 0:
                self.P[0, 0, 0] = 1
                self.P[0, 1, 0] = 0.4
                self.P[0, 1, 1] = 0.6
                self.R[0, 0] = 5. / 1000.
            elif s == n_states - 1:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.4
                self.P[s, 1, s] = 0.6
                self.R[s, 1] = self.r_max
            else:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.05
                self.P[s, 1, s] = 0.6
                self.P[s, 1, s + 1] = 0.35

        super(RiverSwim, self).__init__(states=range(n_states), action_sets=state_actions, P=self.P, gamma=gamma)

    def reset(self, s=0):
        self.state = s
        return self.state

    def step(self, action_index):
        # try:
        #     action_index = self.action_sets[self.state].index(action)
        # except:
        #     raise ValueError("Action {} cannot be executed in this state {}".format(action, self.state))
        p = self.P[self.state, action_index]
        next_state = np.random.choice(self.Ns, 1, p=p).item()
        assert p[next_state] > 0.
        self.reward = self.R[self.state, action_index]
        self.state = next_state
        done = False
        return next_state, self.reward, done, {}

    def true_reward(self, state, action_idx):
        return self.R[state, action_idx]

    def description(self):
        desc = {
            'name': type(self).__name__,
            'nstates': len(self.state_actions)
        }
        return desc

    def sample_transition(self, s, a):
        try:
            p = self.P[s, a]
        except:
            raise ValueError("Action {} cannot be executed in this state {}".format(a, self.state))
        next_state = np.random.choice(self.nb_states, 1, p=p).item()
        return next_state




class ErgodicRiverSwim(FiniteEnv):
    """
    Osband, Van Roy, Russo. (More) Efficient Reinforcement Learning via Posterior Sampling. NIPS

    actions = [0: left, 1: right]
    """

    def __init__(self, n_states=6, gamma=1, uniform_probability=0.001):
        na = 2

        self.r_max = 1.

        self.P = np.zeros((n_states, na, n_states))
        self.R = np.zeros((n_states, na))

        state_actions = []
        for s in range(n_states):
            state_actions.append([0, 1])

            if s == 0:
                self.P[0, 0, 0] = 1
                self.P[0, 0] += uniform_probability
                self.P[0, 1, 0] = 0.4
                self.P[0, 1, 1] = 0.6
                self.P[0, 1] += uniform_probability
                self.R[0, 0] = 5. / 1000.
            elif s == n_states - 1:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.4
                self.P[s, 1, s] = 0.6
                self.P[s, 0] += uniform_probability
                self.P[s, 1] += uniform_probability
                self.R[s, 1] = self.r_max
            else:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.001
                self.P[s, 1, s] = 0.65
                self.P[s, 1, s + 1] = 0.349
                self.P[s, 0] += uniform_probability
                self.P[s, 1] += uniform_probability

            self.P[s, 0] = self.P[s, 0] / self.P[s, 0].sum()
            self.P[s, 1] = self.P[s, 1] / self.P[s, 1].sum()

        assert np.allclose(1, np.sum(self.P, axis=-1))
        super(ErgodicRiverSwim, self).__init__(states=range(n_states), action_sets=state_actions, P=self.P, gamma=gamma)

    def reset(self, s=0):
        self.state = s
        return self.state

    def step(self, action_index):
        # try:
        #     action_index = self.action_sets[self.state].index(action)
        # except:
        #     raise ValueError("Action {} cannot be executed in this state {}".format(action, self.state))
        p = self.P[self.state, action_index]
        next_state = np.random.choice(self.Ns, 1, p=p).item()
        assert p[next_state] > 0.
        self.reward = self.R[self.state, action_index]
        self.state = next_state
        done = False
        return next_state, self.reward, done, {}

    def true_reward(self, state, action_idx):
        return self.R[state, action_idx]

    def description(self):
        desc = {
            'name': type(self).__name__,
            'nstates': len(self.state_actions)
        }
        return desc

    def sample_transition(self, s, a):
        try:
            p = self.P[s, a]
        except:
            raise ValueError("Action {} cannot be executed in this state {}".format(a, self.state))
        next_state = np.random.choice(self.nb_states, 1, p=p).item()
        return next_state
