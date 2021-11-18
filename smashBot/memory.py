from collections import deque
from .models import AgentHyperParameters, Memory 
import json


class Memory:
    def __init__(self, max_memory_len):
        self.states = deque(maxlen=max_memory_len)
        self.actions = deque(maxlen=max_memory_len)
        self.rewards = deque(maxlen=max_memory_len)
        self.next_state = deque(maxlen=max_memory_len)
        self.done = deque(maxlen=max_memory_len)

    def setStates(self, states):
        for state in states:
            self.states.append(state)
    def setActions(self, actions):
        for action in actions:
            self.actions.append(action)
    def setRewards(self, rewards):
        for reward in rewards:
            self.rewards.append(reward)
    def setNextStates(self, next_states):
        for next_state in next_states:
            self.next_state.append(next_state)
    def setDones(self, dones):
        for done in dones:
            self.done.append(done)
