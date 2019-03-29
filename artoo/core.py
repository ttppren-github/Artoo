import os

import gym
import gym.wrappers
import yaml
from easydict import EasyDict

import artoo.logger as logger


def list_envs():
    from gym import envs
    envids = [spec.id for spec in envs.registry.all()]
    for envid in sorted(envids):
        print(envid)


def get_default_conf():
    p = os.path.dirname(__file__)
    f = os.path.join(p, 'config/episode.yaml')
    with open(f) as f:
        return yaml.load(f.read())['episode']


def run(conf, act_func, on_over=None, on_record=None):
    logger.set_level(logger.INFO)

    _conf = get_default_conf()
    if conf:
        _conf.update(conf)
    e_conf = EasyDict(_conf)
    logger.info('Train begin: episode count {}'.format(e_conf.count))

    env = gym.make(e_conf.name)
    if e_conf.enable_monitor:
        env = gym.wrappers.Monitor(env,
                                   os.path.join(os.path.curdir, 'recording'),
                                   force=True)

    if not act_func or not callable(act_func):
        raise RuntimeError('Need a callable action function')

    if on_over and not callable(on_over):
        logger.info('Not set on_over function')

    if on_record and not callable(on_record):
        logger.info('Not set on_record function')

    try:
        for e in range(e_conf.count):
            sts_cur = env.reset()
            for i in range(e_conf.max_step):
                action = act_func(sts_cur)
                sts_next, reword, done, info = env.step(action)
                if on_record:
                    on_record((sts_cur, action, reword, sts_next))
                sts_cur = sts_next

                if e_conf.render.enable:
                    if 0 == e % e_conf.render.interval:
                        env.render()

                if done :
                    if on_over:
                        on_over((sts_cur, action, reword, None))
                    break

            logger.info(f'episode {e}: steps is {i}')
    except KeyboardInterrupt as e:
        logger.warn('KeyboardInterrupt')
    finally:
        env.close()

    logger.info('Train over')
