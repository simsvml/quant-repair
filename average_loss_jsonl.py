import json
import sys

acc = []
lr_acc = []
WINDOW = 50
INTERVAL = 25

i = 0
for line in sys.stdin:
    if line.strip() == '':
        continue
    metrics = json.loads(line)
    loss = metrics['loss']
    lr = metrics['lr']
    if i < 3:
        print('init: loss = %9.3e, lr = %9.3e' % (loss, lr))
    acc.append(loss)
    lr_acc.append(lr)
    assert len(acc) == len(lr_acc)
    if len(acc) > WINDOW:
        del acc[0]
        del lr_acc[0]
        if i % INTERVAL == 0:
            loss_avg = sum(acc) / len(acc)
            lr_avg = sum(lr_acc) / len(lr_acc)
            print('loss = %9.3e, lr = %9.3e' % (loss_avg, lr_avg))
    i += 1


