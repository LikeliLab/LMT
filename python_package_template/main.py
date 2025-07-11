"""This is the python_package_template module.

It provides basic functions for demonstration purposes.
"""


def hello(name: str = 'world') -> str:
    """Return a greeting string.

    Args:
        name (str): The name to greet.

    Returns:
        str: The greeting string.
    """
    return f'Hello {name}'


def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of the two integers.
    """
    return a + b


def multiply(a: float, b: int) -> float:
    """Multiply a float by an integer.

    Args:
        a (float): The float number.
        b (int): The integer number.

    Returns:
        float: The product of the float and the integer.
    """
    return a * b
