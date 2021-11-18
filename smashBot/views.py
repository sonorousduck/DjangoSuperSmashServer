from django.shortcuts import render
from django.http import JsonResponse, FileResponse
import json
import numpy as np
import io
from numpy import unicode
from collections import deque
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Embedding, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import random
from smashBot.agent import Agent
from .models import AgentHyperParameters, Memory 


# Create your views here.


def add_experience(state, action, reward, next_state, done, agent):

    agentMemory = Memory.objects.filter(agent=agent)
    agentMemoryStates = json.loads(agentMemory.states)
    agentMemoryActions = json.loads(agentMemory.actions)
    agentMemoryRewards = json.loads(agentMemory.rewards)
    agentMemoryNextStates = json.loads(agentMemory.next_states)
    agentMemoryDone = json.loads(agentMemory.dones)


    while len(agentMemoryStates) >= agentMemory.max_memory_len:
       agentMemoryStates.pop()
    while len(agentMemoryActions) >= agentMemory.max_memory_len:
       agentMemoryActions.pop()
    while len(agentMemoryRewards) >= agentMemory.max_memory_len:
       agentMemoryRewards.pop()
    while len(agentMemoryNextStates) >= agentMemory.max_memory_len:
       agentMemoryNextStates.pop()
    while len(agentMemoryDone) >= agentMemory.max_memory_len:
       agentMemoryDone.pop()

    agentMemoryStates.insert(0, state)
    agentMemoryActions.insert(0, action)
    agentMemoryRewards.insert(0, reward)
    agentMemoryNextStates.insert(0, next_state)
    agentMemoryDone.insert(0, done)

    agentMemory.states = json.dumps(agentMemoryStates)
    agentMemory.actions = json.dumps(agentMemoryActions)
    agentMemory.rewards = json.dumps(agentMemoryRewards)
    agentMemory.next_states = json.dumps(agentMemoryNextStates)
    agentMemory.dones = json.dumps(agentMemoryDone)

    agentMemory.save()


def create_model():
    model = Sequential()
    model.add(Input(44,))
    model.add(Embedding(44, 128))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(512, activation="swish"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation="swish"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation="swish"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(26, activation="softmax"))
    optimizer = Adam(self.learning_rate)
    model.compile(optimizer, loss='mse')
    return model


def index(request):
    return render(request, "Hello there") 


async def postState(request):
    # This is where the training logic is going to go

    if (request.POST):
        states = request.POST['states']
        actions = request.POST['actions']
        rewards = request.POST['rewards']
        nextStates = request.POST['nextStates']
        dones = request.POST['dones']
        agent = request.POST['agent']
        

        for i in range(len(states)):
            add_experience(states[i], actions[i], rewards[i], nextStates[i], dones[i], agent) 
         
        print("States Added")
        print("Beginning Training")
        model = create_model()
        
        agentMemory = Memory.objects.filter(agent=agent)
        agentHyperparameters = AgentHyperparameters.objects.filter(agent=agent)

        memory = Memory(100000)
        memory.setStates(json.loads(agentMemory.states))
        memory.setActions(json.loads(agentMemory.actions))
        memory.setRewards(json.loads(agentMemory.rewards))
        memory.setNextStates(json.loads(agentMemory.next_states))
        memory.setDones(json.loads(agentmemory.dones))

        minibatch = random.sample(self.memory, 



async def getAgent(request):

    # get the agent from the file
   
    model = open('smashBot/recentWeights.hdf5', 'rb')

    response = FileResponse(model)

    response['Access-Control-Allow-Origin'] = '*'

    return response
    
    #model = {}
    #response = JsonResponse({'model': model})



