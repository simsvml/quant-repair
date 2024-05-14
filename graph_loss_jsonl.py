import json
import sys
import matplotlib.pyplot as plt



loss = []
lr = []

for line in sys.stdin:
    if line.strip() == '':
        continue
    metrics = json.loads(line)
    if 'loss' not in metrics:
        continue
    loss.append(metrics['loss'])
    lr.append(metrics['lr'])

WINDOW_SIZE = 80

loss_avg = []
acc = sum(loss[:WINDOW_SIZE])
loss_avg.append(acc / len(loss[:WINDOW_SIZE]))
for i in range(WINDOW_SIZE, len(loss)):
    acc += loss[i]
    acc -= loss[i - WINDOW_SIZE]
    loss_avg.append(acc / WINDOW_SIZE)


fig, ax = plt.subplots()
#ax.set_ylim((0, 0.5))
#ax.set_ylim((0.05, 0.3))
plt.yscale('log')
plt.grid(visible=True, axis='y')
ax.plot(loss)
ax.plot(range(WINDOW_SIZE//2, WINDOW_SIZE//2 + len(loss_avg)), loss_avg)
plt.show()


