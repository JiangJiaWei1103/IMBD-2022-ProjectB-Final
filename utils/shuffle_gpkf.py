"""
Shuffle GroupKFold cross-validator.

Ref: https://github.com/scikit-learn/scikit-learn/issues/13619
"""
from typing import Iterator, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.utils import check_array, check_random_state


class ShuffleGroupKFold(GroupKFold):
    def __init__(self, n_splits: int = 10, shuffle: bool = True, random_state: int = None):
        super(GroupKFold, self).__init__(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _iter_test_indices(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], groups: Union[pd.Series, np.ndarray]
    ) -> Iterator[np.ndarray]:
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        # Get random state
        rng = check_random_state(self.random_state)

        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )

        # Weight groups by their number of occurrences
        n_samples_per_group = np.bincount(groups)

        # Distribute the most frequent groups first
        indices = np.argsort(n_samples_per_group)[::-1]
        n_samples_per_group = n_samples_per_group[indices]

        # Init already distributed groups
        groups_to_distribute = np.ones(groups.shape, dtype=bool)

        # Total weight of each fold
        n_samples_per_fold = np.zeros(self.n_splits)

        # Mapping from group index to fold index
        group_to_fold = np.zeros(len(unique_groups))

        # Distribute samples by randomly drawing out of the remaining groups for the lightest fold
        while groups_to_distribute.any():
            # take an index from the not distributed indices (intrinsically weighted by relative frequency)
            index = rng.randint(0, np.count_nonzero(groups[groups_to_distribute]), size=None)
            group_to_distribute = groups[groups_to_distribute][index]

            # get lightest fold to fill next
            lightest_fold = np.argmin(n_samples_per_fold)

            # fill lightest fold
            n_samples_per_fold[lightest_fold] += n_samples_per_group[group_to_distribute]
            group_to_fold[group_to_distribute] = lightest_fold
            groups_to_distribute = np.where(groups == group_to_distribute, False, groups_to_distribute)

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]
