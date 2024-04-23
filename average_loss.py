import re
import sys

REGEX = re.compile(r'loss:([0-9.e+-]*)')

acc = []
WINDOW = 500
INTERVAL = 100

i = 0
for line in sys.stdin:
    m = REGEX.search(line)
    if m is not None:
        loss = float(m.group(1))
        if i < 3:
            print('init:', loss)
        acc.append(loss)
        if len(acc) > WINDOW:
            del acc[0]
            if i % INTERVAL == 0:
                print(sum(acc) / len(acc))
        i += 1


