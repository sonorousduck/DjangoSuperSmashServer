from django.shortcuts import render
from django.http import JsonResponse, FileResponse
import json
import numpy as np
import io
import os
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
        
        overallReward = 0
        for reward in rewards:
            overallReward += reward
        print(overallReward)

        for i in range(len(states)):
            add_experience(states[i], actions[i], rewards[i], nextStates[i], dones[i], agent) 
         
        print("States Added")
        print("Beginning Training")
        model = create_model()
        
        agentMemory = Memory.objects.filter(agent=agent)
        agentHyperparameters = AgentHyperparameters.objects.filter(agent=agent)
        shouldSaveAnyways = False

        if not os.path.exists('bestmodel.hdf5'):
            model.load_weights('recentweights.hdf5') 
            agentMemory.bestReward = overallReward
            shouldSaveAnyways = True
        else:
            model.load_weights('bestmodel.hdf5')
        

        memory = Memory(100000)
        memory.setStates(json.loads(agentMemory.states))
        memory.setActions(json.loads(agentMemory.actions))
        memory.setRewards(json.loads(agentMemory.rewards))
        memory.setNextStates(json.loads(agentMemory.next_states))
        memory.setDones(json.loads(agentmemory.dones))

        minibatch = random.sample(self.memory, agentHyperparameters.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []


        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        states = np.asarray(states).astype("float32")
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states).astype("float32")
        dones = np.asarray(dones)

        labels = model.predict(states)
        next_state_values = model.predict(next_states)

        for i in range(agentHyperparameters.batch_size):
            action = actions[i]
            labels[i][action] = rewards[i] + (not dones[i] * agentHyperparameters.gamma * max(next_state_values[i]))

        model.fit(x=states, y=labels, batch_size=agentHyperparameters.batch_size, epochs=10, verbose=1)
        agentHyperparameters.learns += 1


        if agentHyperparameters.epsilon > agentHyperparameters.epsilon_min:
            agentHyperparameters.epsilon *= agentHyperparameters.epsilon_decay
        agentHyperparameters.epsilon = max(agentHyperparameters.epsilon, agentHyperparameters.epsilon_min)

    
        model.save_weights('recentweights.hdf5')

        if overallReward > savedBestReward or shouldSaveAnyways:
            model.save_weights('bestmodel.hdf5')
            agentMemory.bestReward = overallReward

        agentHyperparameters.save()
        agentMemory.save()


async def getAgent(request):

    # get the agent from the file
   
    model = open('smashBot/recentWeights.hdf5', 'rb')

    response = FileResponse(model)

    response['Access-Control-Allow-Origin'] = '*'

    return response


