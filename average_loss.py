import re
import sys

REGEX = re.compile(r'loss:([0-9.e+-]*) lr:([0-9.e+-]*)')

acc = []
lr_acc = []
WINDOW = 100
INTERVAL = 50

i = 0
for line in sys.stdin:
    m = REGEX.search(line)
    if m is not None:
        loss = float(m.group(1))
        lr = float(m.group(2))
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


