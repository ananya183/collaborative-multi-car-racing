import numpy as np

class Car:
    def __init__(self, start_pos, agent_id: int, teammate_id: int):
        self.agent_id = agent_id
        self.teammate_id = teammate_id
        self.position = start_pos
        self.checkpoint_counters = 0
        self.collision_counter = 0
        self.reward = 0
        self.done = False

    def reset(self, position, observation):
        self.position = position
        self.checkpoint_counters = 0
        self.collision_counter = 0
        self.reward = 0
        self.done = False
        self.observation = observation