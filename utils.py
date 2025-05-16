import os
import random
import string


def create_random_exp_tag(directory: str):
    """
    Create a random experiment tag for logging purposes.
    """
    tag = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    if os.path.exists(os.path.join(directory, tag)):
        return create_random_exp_tag(directory)
    return tag
