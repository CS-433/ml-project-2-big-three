from typing import Callable, Any


def print_func_name(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to print function name before executing it.

    :param func: The function to be decorated.
    :type func: Callable[..., Any]

    :return: The decorated function.
    :rtype: Callable[..., Any]
    """

    def wrapper(*args, **kwargs):
        print(f"Executing: `{func.__name__}`")
        return func(*args, **kwargs)

    return wrapper
