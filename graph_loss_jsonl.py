import json
import os
import sys
import matplotlib.pyplot as plt




def read_file(f):
    loss = []
    lr = []

    for line in f:
        if line.strip() == '':
            continue
        metrics = json.loads(line)
        if 'loss' not in metrics:
            continue
        loss.append(metrics['loss'])
        lr.append(metrics['lr'])

    WINDOW_SIZE = 20

    if len(loss) < WINDOW_SIZE:
        return loss, []

    loss_avg = []
    acc = sum(loss[:WINDOW_SIZE])
    loss_avg.append(acc / len(loss[:WINDOW_SIZE]))
    for i in range(WINDOW_SIZE, len(loss)):
        acc += loss[i]
        acc -= loss[i - WINDOW_SIZE]
        loss_avg.append(acc / WINDOW_SIZE)

    return loss, loss_avg


fig, ax = plt.subplots()
#ax.set_ylim((0, 0.5))
#ax.set_ylim((0.05, 0.3))
plt.yscale('log')
plt.grid(visible=True, axis='y')

#loss, loss_avg = read_file(sys.stdin)
#ax.plot(loss)
#ax.plot(range(WINDOW_SIZE//2, WINDOW_SIZE//2 + len(loss_avg)), loss_avg)

for path in sys.argv[1:]:
    loss, loss_avg = read_file(open(path))
    ax.plot(loss_avg, label=os.path.basename(path))
    #ax.plot(loss, label=os.path.basename(path))
    if len(loss_avg) > 0:
        print(os.path.basename(path), loss_avg[-1])

ax.legend()
plt.show()


