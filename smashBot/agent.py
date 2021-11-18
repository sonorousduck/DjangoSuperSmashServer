from smashBot.moveset import Moveset
from smashBot.memory import Memory


class Agent(Moveset):
    def __init__(self, controller):
        Moveset.__init__(self, controller)
        self.memory = deque(maxlen=50000)
        self.controller = controller
        self.moveset = Moveset(controller)
        self.possible_actions = [i for i in range(self.moveset.possibleActions)]
        self.epsilon = 1
        self.epsilon_decay = .9995
        self.epsilon_min = 0.05
        self.gamma = 0.90
        self.learning_rate = 0.0025
        self.batch_size = 256
        self.learns = 0
        self.model = self.create_model()
        self.target_model = clone_model(self.model)
        self.rewards = []
        self.averageRewardList = []
        self.oneReward = 0
        self.model = self.create_model()


    def create_model(self):
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
        model.add(Dense(self.moveset.possibleActions, activation="softmax"))
        optimizer = Adam(self.learning_rate)
        model.compile(optimizer, loss='mse')
        model.summary()
        return model


    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions, 1)[0]

        return self.possible_actions[np.argmax(self.model.predict(state))]


    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for state, action, reward, next_state, done in minibatch:

            # for i in state:
            #     print(type(i))
                # x = list(i)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            # for i in next_state:
            #     try:
            #         x = list(i)
            #         next_states.extend(x)
            #     except:
            #         x = np.asarray(i).astype("float32")
            #         next_states.extend(x)
                # next_states.extend(x)
            next_states.append(next_state)
            dones.append(done)
            self.oneReward += reward
        states = np.asarray(states).astype("float32")
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states).astype("float32")
        dones = np.asarray(dones)

        labels = self.model.predict(states)
        next_state_values = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            action = self.possible_actions.index(actions[i])
            labels[i][action] = rewards[i] + (not dones[i]) * self.gamma * max(next_state_values[i])

        self.model.fit(x=states, y=labels, batch_size=self.batch_size, epochs=1, verbose=1)


        self.learns += 1
        if self.learns % 10000 == 0:
            self.target_model.set_weights(self.model.get_weights())
            print('\nTarget model updated')


    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        if len(self.memory) > self.batch_size:
            self.rewards.append(self.oneReward)

            if len(self.rewards) > 250:
                self.averageRewardList.append(np.mean(self.rewards[:-250]))
            else:
                self.averageRewardList.append(np.mean(self.rewards))

        self.oneReward = 0


    def remember(self, state, next_state, action, reward, done):
        self.memory.append((state, action, reward, next_state, done))
