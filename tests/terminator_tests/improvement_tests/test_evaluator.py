from __future__ import annotations

from unittest import mock

import pytest

from optuna.distributions import FloatDistribution
from optuna.study import StudyDirection
from optuna.terminator.improvement._preprocessing import NullPreprocessing
from optuna.terminator.improvement.evaluator import RegretBoundEvaluator
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


# TODO(g-votte): test the following edge cases
# TODO(g-votte): - the user specifies non-default top_trials_ratio or min_n_trials


@pytest.mark.parametrize(
    "trials",
    [
        [
            create_trial(
                value=1.0,
                distributions={
                    "bacon": FloatDistribution(-1.0, 1.0),
                    "egg": FloatDistribution(-1.0, 1.0),
                },
                params={
                    "bacon": 1.0,
                    "egg": 0.0,
                },
            ),
            create_trial(
                value=-1.0,
                distributions={
                    "bacon": FloatDistribution(-1.0, 1.0),
                    "egg": FloatDistribution(-1.0, 1.0),
                },
                params={
                    "bacon": 0.0,
                    "egg": 1.0,
                },
            ),
        ],
        [
            create_trial(
                value=1.0,
                distributions={
                    "bacon": FloatDistribution(-1.0, 1.0),
                    "egg": FloatDistribution(-1.0, 1.0),
                },
                params={
                    "bacon": 1.0,
                    "egg": 0.0,
                },
            ),
        ],
        [
            create_trial(
                value=1.0,
                distributions={
                    "spam": FloatDistribution(-1.0, 1.0),
                },
                params={
                    "spam": 1.0,
                },
            ),
        ],
        [
            create_trial(
                value=1.0,
                distributions={
                    "bacon": FloatDistribution(-1.0, 1.0),
                    "egg": FloatDistribution(-1.0, 1.0),
                },
                params={
                    "bacon": 1.0,
                    "egg": 0.0,
                },
            ),
            create_trial(
                value=1.0,
                distributions={
                    "bacon": FloatDistribution(-1.0, 1.0),
                    "egg": FloatDistribution(-1.0, 1.0),
                },
                params={
                    "bacon": 1.0,
                    "egg": 0.0,
                },
            ),
        ],
    ],
)
def test_evaluate(trials: list[FrozenTrial]) -> None:
    trials = [
        create_trial(
            value=0,
            distributions={"a": FloatDistribution(-1.0, 1.0)},
            params={"a": 0.0},
        )
    ]

    # The purpose of the following mock scope is to maintain loose coupling between the tests for
    # preprocessing and those for the `RegretBoundEvaluator` class. The preprocessing logic is
    # thoroughly tested in another file:
    # tests/terminator_tests/improvement_tests/test_preprocessing.py.
    with mock.patch.object(
        RegretBoundEvaluator, "get_preprocessing", return_value=NullPreprocessing()
    ):
        evaluator = RegretBoundEvaluator()
        evaluator.evaluate(trials, study_direction=StudyDirection.MAXIMIZE)


def test_evaluate_with_no_trial() -> None:
    evaluator = RegretBoundEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(trials=[], study_direction=StudyDirection.MAXIMIZE)


def test_evaluate_with_empty_intersection_search_space() -> None:
    evaluator = RegretBoundEvaluator()

    trials = [
        create_trial(
            value=0,
            distributions={},
            params={},
        )
    ]

    with pytest.raises(ValueError):
        evaluator.evaluate(trials=trials, study_direction=StudyDirection.MAXIMIZE)
