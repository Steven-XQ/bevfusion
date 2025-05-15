import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import argparse
import numpy as np

EPOCH_ITER = 15448
LOG_INTERVAL = 50
NUM_TEXT_LABELS = 15

parser = argparse.ArgumentParser()
parser.add_argument('infile', type=str)
args = parser.parse_args()

outfile = '/'.join(args.infile.split('/')[:-1]) + '/training_dynamics.png'

dicts = []
with open(args.infile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        dicts.append(json.loads(line))

epoch = []
mAP = []
nds = []

iteration = []
loss = []

for dict in dicts:
    if dict:
        if dict['mode'] == 'val':
            epoch.append(dict['epoch'] + 1)
            mAP.append(dict['object/map'])
            nds.append(dict['object/nds'])
        else:
            iteration.append((dict['epoch']) * EPOCH_ITER + dict['iter'])
            loss.append(dict['loss'])

loss = np.array(loss)
iteration = np.array(iteration)[loss < 5]
loss = loss[loss < 5]

plt.figure(figsize=(len(epoch), 10))
gs = gridspec.GridSpec(2, 2)

ax0 = plt.subplot(gs[0, :])
ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
ax0.plot(iteration, loss, label='loss')
epoch_ends = list(range(EPOCH_ITER, iteration[-1] + 1, EPOCH_ITER))
xmax = ax0.get_xlim()[1]
ymax = ax0.get_ylim()[1]
ax0.set_ylim(top=ymax * 1.05)
ax0.axvline(0, color='gray', linestyle='--', linewidth=1)
for i, x in enumerate(epoch_ends):
    ax0.axvline(x, color='gray', linestyle='--', linewidth=1)
    ax0.text(x - EPOCH_ITER / 2, ymax, f'Epoch {i + 1}', ha='center', color='gray')
for x, y in zip(iteration[::int(xmax // (LOG_INTERVAL * NUM_TEXT_LABELS))], loss[::int(xmax // (LOG_INTERVAL * NUM_TEXT_LABELS))]):
    ax0.text(x, y + ymax * 0.03, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
ax0.set_xlabel('Iteration')
ax0.set_ylabel('Loss')
ax0.set_title('Training Loss')

ax1 = plt.subplot(gs[1, 0])
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.plot(epoch, mAP, marker='o', label='mAP')
for x, y in zip(epoch, mAP):
    ax1.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('mAP')
ax1.set_title('mAP')

ax2 = plt.subplot(gs[1, 1])
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.plot(epoch, nds, marker='o', label='NDS')
for x, y in zip(epoch, nds):
    ax2.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('NDS')
ax2.set_title('NDS')

plt.tight_layout()
plt.savefig(outfile)
