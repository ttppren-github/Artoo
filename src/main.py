#!/usr/bin/envs python
import argparse
import inspect
import importlib

import yaml
from easydict import EasyDict

from envs import Env
from base.model_base import ModelBase
from base.train_base import TrainBase


def _import_module(base, path, model_name):
    print(f'load model: {model_name}')
    assert(model_name is not None)

    imported_module = importlib.import_module(path + model_name, None)
    for i in dir(imported_module):
        attribute = getattr(imported_module, i)

        if inspect.isclass(attribute) and issubclass(attribute, base) and attribute is not base:
            return attribute

    return None

def train(conf):
    agent_func = _import_module(ModelBase, 'models.', conf['model']['name'])
    agent  = agent_func(EasyDict(conf['model']))
    coach_func = _import_module(TrainBase, 'train.', conf['train']['name'])
    coach = coach_func(agent, EasyDict(conf['train']))

    assert(agent is not None)
    assert(coach is not None)

    envs = []
    for i in range(conf['env'].get('count', 1)):
        envs.append(Env(agent, coach, EasyDict(conf['env'])))

    envs[0].play()
    envs[0].close()


def evaluate(conf):
    agent_func = _import_module(ModelBase, 'models.', conf['model']['name'])
    agent  = agent_func(EasyDict(conf['model']))

    envs = []
    for i in range(conf['env'].get('count', 1)):
        envs.append(Env(agent, None, EasyDict(conf['env'])))

    envs[0].play()
    envs[0].close()


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--train-conf', help='yaml configure file for training')
    argp.add_argument('--evaluate-conf', help='yaml configure file for evaluate')
    args = argp.parse_args()

    if args.train_conf:
        with open(args.train_conf, 'r') as f:
            conf = yaml.load(f)
        train(conf)
    elif args.evaluate_conf:
        with open(args.evaluate_conf, 'r') as f:
            conf = yaml.load(f)
        evaluate(conf)
    else:
        argp.print_usage()

    print('bye bye')
