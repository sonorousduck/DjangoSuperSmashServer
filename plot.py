import matplotlib.pyplot as plt
import numpy as np

y1 = []
# y2 = []

with open('rewards2.txt') as f:
    while True:
        next_line = f.readline()
        if not next_line:
            break
        y1.append(float(next_line.strip()))


# with open('smashBot/rewards2.txt') as f:
#     while True:
#         next_line = f.readline()
#         if not next_line:
#             break
#         y2.append(float(next_line.strip()))
print(len(y1))
plt.plot(np.arange(0, len(y1)), y1, label='agent')
# plt.plot(np.arange(0, len(y2)), y2, label='agent 2')
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.title("Rewards over Epochs")
plt.legend()
plt.show()