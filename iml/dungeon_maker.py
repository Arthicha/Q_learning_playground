import os
import numpy as np

from iml.dungeon import Dungeon


def make_treasure_room(**kwargs):
        return Dungeon("""####
#s.#
#.e#
####""", **kwargs)#, p_slippery=p_slippery)

def make_underground_maze(p_slippery=0.0):
    return Dungeon("""
#######
#s....#
#.#####
#...e.#
#######
""", p_slippery=p_slippery)

def make_underground_maze_with_meat(p_slippery=0.0):
    return Dungeon("""
#######
#s..+.#
#.#####
#...e.#
#######
""", p_slippery=p_slippery)

def make_dragons_den(**kwargs):
    return Dungeon("""
###oooo#
#s....e#
#.#oooo#
#......#
#.....e#
########
""", **kwargs, )