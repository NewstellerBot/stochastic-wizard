import numpy as np
import scipy.stats as sts


def calculate_confidence_interval(a: np.array, ci: float, n_trials: int):

    return np.array(
        [
            sts.t.interval(
                ci,
                n_trials - 1,
                loc=mean,
                scale=(std + 1e-7) / np.sqrt(n_trials),
            )
            for mean, std in zip(a.mean(0), a.std(0))
        ]
    )
