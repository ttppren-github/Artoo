from .random import Random


def create_agent(name, sts_dim, act_dim):
    if 'random' == name:
        return Random(sts_dim, act_dim)
    else:
        raise RuntimeError('Invalid name:' + name)
