import json
import os
import sys
import matplotlib.pyplot as plt


def exp_smooth(xs, alpha = 0.2):
    cur = None
    for x in xs:
        if cur is None:
            cur = x
        else:
            cur = alpha * x + (1 - alpha) * cur
        yield cur

def sliding_window(xs, size):
    if len(xs) < size:
        return []

    avg = []
    acc = sum(xs[:size])
    avg.append(acc / len(xs[:size]))
    for i in range(size, len(xs)):
        acc += xs[i]
        acc -= xs[i - size]
        avg.append(acc / size)
    return avg

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

    return loss


fig, ax = plt.subplots()
#ax.set_ylim((0, 0.5))
#ax.set_ylim((0.05, 0.3))
plt.yscale('log')
plt.grid(visible=True, axis='y')

#loss, loss_avg = read_file(sys.stdin)
#ax.plot(loss)
#ax.plot(range(WINDOW_SIZE//2, WINDOW_SIZE//2 + len(loss_avg)), loss_avg)

for path in sys.argv[1:]:
    loss = read_file(open(path))
    if len(loss) == 0:
        continue
    #loss2 = list(exp_smooth(loss, alpha = 0.05))
    #loss3 = list(exp_smooth(loss2, alpha = 0.01))
    loss2 = list(sliding_window(loss, size = 250))
    #ax.plot(loss, label=os.path.basename(path))
    ax.plot(loss2, label=os.path.basename(path))
    #ax.plot(loss3, label=os.path.basename(path))
    print(os.path.basename(path), loss2[-1])
    #ax.plot(loss, label=os.path.basename(path))

ax.legend()
fig.set_tight_layout(True)
fig.savefig('graph.png')
plt.show()


