# -*- coding: utf-8 -*-
import Gym

__games = {'Cartpole': 'CartPole-V0'}

def create_world(name):
    world = None
    if name in __games.keys():
        world = Gym(__games[name])

    return world