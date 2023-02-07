from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
from typing import Sequence

import numpy as np
import pytest

import optuna
from optuna.storages import BaseStorage
from optuna.study import StudyDirection
from optuna.trial import TrialState


_STUDY_NAME = "_test_multiprocess"


def f(x: float, y: float) -> float:
    return (x - 3) ** 2 + y


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    trial.report(x, 0)
    trial.report(y, 1)
    trial.set_user_attr("x", x)
    return f(x, y)


def get_storage() -> BaseStorage:
    if "TEST_DB_URL" not in os.environ:
        pytest.skip("This test requires TEST_DB_URL.")
    storage_url = os.environ["TEST_DB_URL"]
    storage_mode = os.environ.get("TEST_DB_MODE", "")

    storage: BaseStorage
    if storage_mode == "":
        storage = optuna.storages.RDBStorage(url=storage_url)
    elif storage_mode == "journal-redis":
        journal_redis_storage = optuna.storages.JournalRedisStorage(storage_url)
        storage = optuna.storages.JournalStorage(journal_redis_storage)
    else:
        assert False, f"The mode {storage_mode} is not supported."

    return storage


@pytest.fixture
def storage() -> BaseStorage:
    storage = get_storage()
    try:
        optuna.study.delete_study(study_name=_STUDY_NAME, storage=storage)
    except KeyError:
        pass
    return storage


def start_trace():
    import sys
    import inspect
    import os
    pid = os.getpid()
    
    def tracer(frame, event, arg, depth=[0]):
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        f_name = frame.f_code.co_name
        # if f_name == '_ag':  # dirty hack for CPython 3.6+
        #     return

        if event == 'call':
            depth[0] += 1
            try:
                args = inspect.formatargvalues(*inspect.getargvalues(frame))
            except:
                args = "<<<Error!>>>"
            print(pid, filename, lineno, '>' * depth[0], f_name, args)
        elif event == 'return':
            try:
                ret = arg.__repr__()
            except:
                ret = "<<<Error!>>>"
            print(pid, filename, lineno, '<' * depth[0], f_name, ret)
            depth[0] -= 1
        return tracer

    sys.settrace(tracer)

import time
import threading
import os
import signal
import sys
import traceback
import io

def run_optimize(study_name: str, storage: BaseStorage, n_trials: int) -> None:
    pid = os.getpid()
    try:
        print(f"{pid}> hoge")
        # raise Exception("aaa")
        # start_trace()
        # Create a study
        latest = time.monotonic()
        study = optuna.load_study(study_name=study_name, storage=storage)
        stop = False
        def tick():
            cur = time.monotonic()
            if cur - latest > 20:
                print(f"{pid}> *** Deadlock detected ***")
                os.kill(os.getpid(), signal.SIGINT)
            if stop:
                return
            
            timer = threading.Timer(1, function=tick)
            timer.start()
        
        # thread = threading.Thread(target=run)
        timer = threading.Timer(1, function=tick)
        timer.start()
        # Run optimization
        def callback(*args):
            nonlocal latest
            print(f"{pid}> update {latest}")
            latest = time.monotonic()
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        stop=True
    except (Exception, KeyboardInterrupt) as e:
        strio = io.StringIO() #newline=f"\n{pid}> ")
        traceback.print_exception(e, file=strio)
        lines = strio.getvalue().splitlines()
        out = "\n".join([f"{pid}> {s}" for s in lines])
        print(out)
        raise e

def _check_trials(trials: Sequence[optuna.trial.FrozenTrial]) -> None:
    # Check trial states.
    assert all(trial.state == TrialState.COMPLETE for trial in trials)

    # Check trial values and params.
    assert all("x" in trial.params for trial in trials)
    assert all("y" in trial.params for trial in trials)
    assert all(
        np.isclose(
            np.asarray([trial.value for trial in trials]),
            [f(trial.params["x"], trial.params["y"]) for trial in trials],
            atol=1e-4,
        ).tolist()
    )

    # Check intermediate values.
    assert all(len(trial.intermediate_values) == 2 for trial in trials)
    assert all(trial.params["x"] == trial.intermediate_values[0] for trial in trials)
    assert all(trial.params["y"] == trial.intermediate_values[1] for trial in trials)

    # Check attrs.
    assert all(
        np.isclose(
            [trial.user_attrs["x"] for trial in trials],
            [trial.params["x"] for trial in trials],
            atol=1e-4,
        ).tolist()
    )


# def test_tracer():
#     start_trace()
#     def fib(i):
#         if i <= 1:
#             return 1
#         return fib(i - 1) + fib(i - 2)
#     assert fib(5) == 8
# def test_loaded_trials(storage: BaseStorage) -> None:
#     print("test_loaded_trials 開始")
#     # Please create the tables by placing this function before the multi-process tests.

#     N_TRIALS = 20
#     study = optuna.create_study(study_name=_STUDY_NAME, storage=storage)
#     # Run optimization
#     study.optimize(objective, n_trials=N_TRIALS)

#     trials = study.trials
#     assert len(trials) == N_TRIALS

#     _check_trials(trials)

#     # Create a new study to confirm the study can load trial properly.
#     loaded_study = optuna.load_study(study_name=_STUDY_NAME, storage=storage)
#     _check_trials(loaded_study.trials)
#     print("test_loaded_trials 終了")


# @pytest.mark.parametrize(
#     "input_value,expected",
#     [
#         (float("inf"), float("inf")),
#         (-float("inf"), -float("inf")),
#     ],
# )
# def test_store_infinite_values(input_value: float, expected: float, storage: BaseStorage) -> None:
#     print("test_store_infinite_values 開始")
#     study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
#     trial_id = storage.create_new_trial(study_id)
#     storage.set_trial_intermediate_value(trial_id, 1, input_value)
#     storage.set_trial_state_values(trial_id, state=TrialState.COMPLETE, values=(input_value,))
#     assert storage.get_trial(trial_id).value == expected
#     assert storage.get_trial(trial_id).intermediate_values[1] == expected
#     print("test_store_infinite_values 終了")


# def test_store_nan_intermediate_values(storage: BaseStorage) -> None:
#     print("test_store_nan_intermediate_values 開始")
#     study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
#     trial_id = storage.create_new_trial(study_id)

#     value = float("nan")
#     storage.set_trial_intermediate_value(trial_id, 1, value)

#     got_value = storage.get_trial(trial_id).intermediate_values[1]
#     assert np.isnan(got_value)
#     print("test_store_nan_intermediate_values 終了")


# def test_multithread_create_study(storage: BaseStorage) -> None:
#     print("test_multithread_create_study 開始")
#     with ThreadPoolExecutor(10) as pool:
#         for _ in range(10):
#             pool.submit(
#                 optuna.create_study,
#                 storage=storage,
#                 study_name="test-multithread-create-study",
#                 load_if_exists=True,
#             )
#     print("test_multithread_create_study 終了")


def test_multiprocess_run_optimize(storage: BaseStorage) -> None:
    print("test_multiprocess_run_optimize 開始")
    n_workers = 8
    n_trials = 20
    study_name = _STUDY_NAME
    optuna.create_study(storage=storage, study_name=study_name)
    with ProcessPoolExecutor(n_workers) as pool:
        pool.map(run_optimize, *zip(*[[study_name, storage, n_trials]] * n_workers))

    study = optuna.load_study(study_name=study_name, storage=storage)

    trials = study.trials
    assert len(trials) == n_workers * n_trials

    _check_trials(trials)
    print("test_multiprocess_run_optimize 終了")


# def test_pickle_storage(storage: BaseStorage) -> None:
#     print("test_pickle_storage 開始")
#     study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
#     storage.set_study_system_attr(study_id, "key", "pickle")

#     restored_storage = pickle.loads(pickle.dumps(storage))

#     storage_system_attrs = storage.get_study_system_attrs(study_id)
#     restored_storage_system_attrs = restored_storage.get_study_system_attrs(study_id)
#     assert storage_system_attrs == restored_storage_system_attrs == {"key": "pickle"}
#     print("test_pickle_storage 終了")
