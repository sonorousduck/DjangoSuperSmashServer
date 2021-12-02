import threading
import time
from collections import deque

from django.shortcuts import render
from django.http import JsonResponse, Http404, HttpResponse, FileResponse
import json
import numpy as np
import os

from django.views.decorators.csrf import csrf_exempt
from keras.layers import Flatten
from requests import Response, status_codes
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Embedding, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import random
from .memory import AgentMemory
from .models import AgentHyperparameters, Memory
from schedule import Scheduler

# Create your views here.

class ResponseThen(Response):
    def __init__(self, data, then_callback, **kwargs):
        super().__init__(data, **kwargs)
        self.then_callback = then_callback

    def close(self):
        super().close()
        self.then_callback()



def add_experience(state, action, reward, next_state, done, agent):
    agentMemory = Memory.objects.get(agent=agent)
    agentMemoryStates = json.loads(agentMemory.states)
    agentMemoryActions = json.loads(agentMemory.actions)
    agentMemoryRewards = json.loads(agentMemory.rewards)
    agentMemoryNextStates = json.loads(agentMemory.next_states)
    agentMemoryDone = json.loads(agentMemory.dones)

    for i in range(len(state)):
        agentMemoryStates.insert(0, state[i])
        agentMemoryActions.insert(0, action[i])
        agentMemoryRewards.insert(0, reward[i])
        agentMemoryNextStates.insert(0, next_state[i])
        agentMemoryDone.insert(0, done[i])


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


    agentMemory.states = json.dumps(agentMemoryStates)
    agentMemory.actions = json.dumps(agentMemoryActions)
    agentMemory.rewards = json.dumps(agentMemoryRewards)
    agentMemory.next_states = json.dumps(agentMemoryNextStates)
    agentMemory.dones = json.dumps(agentMemoryDone)

    agentMemory.save()


def create_model():
    model = Sequential()
    model.add(Input(26,))
    #model.add(Embedding(44, 128))
    #model.add(LSTM(128, return_sequences=True))
    #model.add(LSTM(256))
    model.add(Dense(128, activation="swish"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation="swish"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation="swish"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation="swish"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(30, activation="softmax"))
    optimizer = Adam(0.00025)
    model.compile(optimizer, loss='mse')
    # model.summary()
    return model



def index(request):
    return render(request, "smashBot/index.html")


def test(request):
    for i in range(50):
        train(2, 1000)
    json_response = [{'success': "Success!"}]
    response = JsonResponse(json_response, safe=False)
    response['Access-Control-Allow-Origin'] = '*'
    return response

def train(agent, overallReward):
    # Add in agent and overall reward for both
    print("States Added")
    print("Beginning Training")
    model = create_model()
    agentMemory = Memory.objects.get(agent=agent)
    agentHyperparameters = AgentHyperparameters.objects.get(agent=agent)
    shouldSaveAnyways = False

    if not os.path.exists('bestweights.hdf5'):
        if os.path.exists('recentweights.hdf5'):
            model.load_weights('recentweights.hdf5')
            agentMemory.bestReward = overallReward
        shouldSaveAnyways = True
    else:
        model.load_weights('bestweights.hdf5')

    states = json.loads(agentMemory.states)
    actions = json.loads(agentMemory.actions)
    rewards = json.loads(agentMemory.rewards)
    nextStates = json.loads(agentMemory.next_states)
    dones = json.loads(agentMemory.dones)


    memoryDeque = deque(maxlen=250000)

    for i in range(len(states)):
        memoryDeque.append((states[i], actions[i], rewards[i], nextStates[i], dones[i]))

    batch_size = 128 
    minibatch = random.sample(memoryDeque, batch_size)
    # states = []
    # actions = []
    # rewards = []
    # next_states = []
    # dones = []
    everyTarget = []
    everyState = []

    for state, action, reward, next_state, done in minibatch:
        target = reward

        if not done:
            # x = np.amax(model.predict(next_state))
            # y_state = model.predict(state)
            # y = y_state[action]
            # test = model.predict(next_state)
            # test1 = test[0]
            # print(np.max(model.predict(next_state)[0]))
            # print(model.predict(state)[action])
            next_state = np.array(next_state)
         #   print(next_state.shape)
            next_state = next_state.reshape(1, -1)
        #    print(next_state.shape)

            target = reward + float(agentHyperparameters.gamma) * np.max(model.predict(next_state))

        state = np.array(state)
        state = state.reshape(1, -1)
        target_f = model.predict(np.array(state))[0]
        target_f[action] = target
        target_f = target_f.reshape(1, -1)
        #everyTarget.append(target_f)
        #everyState.append(state)


        # print()
        # print(type(state))
        # print(type(target_f))
        # print(target_f)

        model.fit(state, target_f, epochs=1, verbose=1)


        # If Q-learning doesn't work well, use SARSA. I definitely wasn't doing either of these very correctly
        # Q_s_t = model.predict(state)
        #target = Q_s_t + agentHyperparameters.learning_rate(reward + float(agentHyperparameters.gamma) * model.predict(next_state) - Q_s_t)
            

       # states.append(state)
       # actions.append(action)
       # rewards.append(reward)
       # next_states.append(next_state)
       # dones.append(done)
    #model.fit(everyState, everyTarget, batch_size=1, epochs=10, verbose=1)
    #states = np.asarray(states).astype("float32")
    #actions = np.asarray(actions)
    #rewards = np.asarray(rewards)
    #next_states = np.asarray(next_states).astype("float32")
    #dones = np.asarray(dones)

    # labels = model.predict(states)
    # next_state_values = model.predict(next_states)

    # batchSize = min(agentHyperparameters.batch_size, len(rewards))
    # for i in range(batchSize):
     #   action = actions[i]
     #   labels[i][action] = rewards[i] + (
     #       not dones[i] * float(agentHyperparameters.gamma) * max(next_state_values[i]))

    # for i in range(len(states)):
    #
    # model.fit(x=states, y=labels, epochs=1, verbose=1)
    agentHyperparameters.learns += 1

    if agentHyperparameters.epsilon > agentHyperparameters.epsilon_min:
        agentHyperparameters.epsilon *= agentHyperparameters.epsilon_decay
    agentHyperparameters.epsilon = max(agentHyperparameters.epsilon, agentHyperparameters.epsilon_min)

    model.save_weights('recentweights.hdf5')

    if overallReward > agentHyperparameters.bestReward or shouldSaveAnyways:
        print(f"Saving model from agent {agentHyperparameters.agent}")
        model.save_weights('bestweights.hdf5')
        agentHyperparameters.bestReward = overallReward
    
    agentHyperparameters.save()
    agentMemory.save()

    json_response = [{'success': "Success!"}]
    response = JsonResponse(json_response, safe=False)
    response['Access-Control-Allow-Origin'] = '*'
    return response


@csrf_exempt
def postState(request):
    # This is where the training logic is going to go
    # print(request)
    data = json.loads(request.body)

    if (request.method == 'POST'):

        states = data['states']
        actions = data['actions']
        rewards = data['rewards']
        nextStates = data['nextStates']
        dones = data['dones']
        agent = data['agent']

        overallReward = 0
        for reward in rewards:
            overallReward += reward
        print(overallReward)

        with open(f'smashBot/rewards{agent}.txt', 'a') as f:
            f.write(str(overallReward) + '\n')


        add_experience(states, actions, rewards, nextStates, dones, agent)
        train(agent, overallReward)

        json_response = [{'success': "Success!"}]
        response = JsonResponse(json_response, safe=False)
        response['Access-Control-Allow-Origin'] = '*'
        return response
    json_response = [{'failure': "Failed!"}]
    response = JsonResponse(json_response, safe=False)
    response['Access-Control-Allow-Origin'] = '*'

    return response


async def getAgent(request):
    # get the agent from the file

    if not os.path.exists('bestweights.hdf5'):
        if not os.path.exists('recentweights.hdf5'):
            raise Http404("File not yet created")

        model = open('recentweights.hdf5', 'rb')
        response = FileResponse(model)
        response['Access-Control-Allow-Origin'] = '*'
    else:
        model = open('bestweights.hdf5', 'rb')
        response = FileResponse(model)
        response['Access-Control-Allow-Origin'] = '*'

    return response
