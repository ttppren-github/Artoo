from worlds import *
from agent.randam_agent import RandomAgent

def train():
    pass

def test():
    print(worlds.world)
    game = worlds.world.create_world('cartpole')
    game.enable_render = True

    agent = RandomAgent(game.state_dim, game.action_dim)

    game.reset()
    for episode in range(10):
        for n in range(300):
            act = agent.act(game.stat)
            game.step(act)

            if game.terminal:
                print("Episode {} terminal, steps is {}".format(episode, n))
                break
    print('Test over')


def main():
    test() 

if __name__ == '__main__':
    main()
