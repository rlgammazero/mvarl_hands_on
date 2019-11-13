import numpy as np


class LinearMABModel(object):
    def __init__(self, random_state=0, noise=0.1, features=None, theta=None):
        self.local_random = np.random.RandomState(random_state)
        self.noise = noise
        self.features = features
        self.theta = theta

    def reward(self, action):
        assert 0 <= action < self.n_actions, "{} not in 0 .. {}".format(action, self.n_actions)
        reward = np.dot(self.features[action], self.theta) + self.noise * self.local_random.randn(1)
#        mean = np.dot(self.features[action], self.theta)
#        reward = np.random.binomial(1, mean)

        return reward

    def best_arm_reward(self):
        D = np.dot(self.features, self.theta)
        return np.max(D)

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_actions(self):
        return self.features.shape[0]


class RandomLinearArms(LinearMABModel):
    def __init__(self, random_state=0, noise=0., n_actions=4, n_features=4, bound_features=1, bound_theta = 1, positive=True, max_one=True):
        features = np.random.randn(n_actions, n_features)
        real_theta = np.random.randn(n_features)
        real_theta = np.random.uniform(low = 1/2, high = bound_theta)*real_theta/np.linalg.norm(real_theta)
        if positive:
            idxs = np.dot(features, real_theta) <= 0
            idxs = np.arange(n_actions)[idxs]
            for i in idxs:
                mean = -1
                feat = None
                while mean <= 0:
                    feat = np.random.randn(1, n_features)
                    mean = np.dot(feat, real_theta)
                features[i, :] = feat
        features = np.random.uniform(low = 1/2, high = bound_features, size = (n_actions,1)) * features / max(np.linalg.norm(features, axis=1))

        if max_one:
            D = np.dot(features, real_theta)

            min_rwd = min(D)
            max_rwd = max(D)
            min_features = features[np.argmin(D)]
            features = (features - min_features) / (max_rwd - min_rwd)

        super(RandomLinearArms, self).__init__(random_state=random_state, noise=noise,
                                               features=features, theta=real_theta)


class ColdStartFromDataset(LinearMABModel):
    def __init__(self, arm_csvfile, user_csvfile, random_state=0, noise_std=0., user_subset=None):
        features = np.loadtxt(arm_csvfile, delimiter=',').T
        thetas = np.loadtxt(user_csvfile, delimiter=',')

        super(ColdStartFromDataset, self).__init__(random_state=random_state, noise=noise_std,
                                                        features=features, theta=None)
        if user_subset is None:
            self.theta_idx = np.random.randint(low=0, high=thetas.shape[0])
        else:
            self.theta_idx = np.random.choice(user_subset)
        # print("Selecting user: {}".format(self.theta_idx))
        self.theta = thetas[self.theta_idx]
        # self.theta = np.random.randn(thetas.shape[1])

        D = np.dot(self.features, self.theta)

        min_rwd = min(D)
        max_rwd = max(D)
        min_features = features[np.argmin(D)]
        self.features = (self.features - min_features) / (max_rwd - min_rwd)
