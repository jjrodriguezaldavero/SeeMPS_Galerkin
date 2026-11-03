import numpy as np
import pickle
import os
import multiprocessing as mp
from contextlib import nullcontext
from functools import wraps
from threadpoolctl import threadpool_limits
from typing import Any

from seemps_galerkin.typing import Vector


def save_pkl(data: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pkl(path: str) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)


def get_distance_Lp(
    u_exact: Vector, u_approx: Vector, p: float = 2, relative: bool = False
) -> float:
    error = np.linalg.norm((u_exact - u_approx).ravel(), ord=p)
    if relative:
        error /= np.linalg.norm(u_exact.ravel(), ord=p)
    return float(error)


def fresh_process(**env):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ctx = mp.get_context("fork")
            parent_conn, child_conn = ctx.Pipe(duplex=False)

            def worker():
                os.environ.update(env)
                try:
                    limit = env.get("OPENBLAS_NUM_THREADS")
                    context = (
                        nullcontext() if limit is None else threadpool_limits(limits=int(limit))
                    )
                    with context:
                        result = func(*args, **kwargs)
                    child_conn.send((True, result))
                except Exception as exc:  # noqa
                    child_conn.send((False, exc))
                finally:
                    child_conn.close()

            process = ctx.Process(target=worker)
            process.start()
            child_conn.close()
            try:
                success, result = parent_conn.recv()
            except EOFError as exc:
                process.join()
                raise RuntimeError(
                    f"Worker exited without a result (exit code {process.exitcode})"
                ) from exc
            finally:
                parent_conn.close()
            process.join()

            if not success:
                raise result

            return result

        return wrapper

    return decorator
