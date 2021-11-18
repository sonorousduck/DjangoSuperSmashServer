from django.db import models
import json
# Create your models here.

class AgentHyperparameters(models.Model):
    epsilon = models.DecimalField(max_digits=20, decimal_places=18)
    epsilon_decay = models.DecimalField(max_digits=10, decimal_places=8, default=0.9995)
    epsilon_min = models.DecimalField(max_digits=10, decimal_places=8, default=0.05)
    gamma = models.DecimalField(max_digits=5, decimal_places=4, default=0.90)
    learning_rate = models.DecimalField(max_digits=10, decimal_places=9, default=0.0025)
    batch_size = models.IntegerField(default=256)
    learns = models.IntegerField(default=0)
    averageRewardList = models.JSONField(default={})
    agent = models.IntegerField()
    bestReward = models.DecimalField(max_digits=40, decimal_places=20)


class Memory(models.Model):
    max_memory_len = models.IntegerField(default=100000)
    agent = models.IntegerField()
    states = models.JSONField(default={})
    actions = models.JSONField(default={})
    rewards = models.JSONField(default={})
    next_states = models.JSONField(default={})
    dones = models.JSONField(default={})

