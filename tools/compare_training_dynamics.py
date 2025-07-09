import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import argparse
import numpy as np

EPOCH_ITER = 7724
LOG_INTERVAL = 100
NUM_TEXT_LABELS = 15

parser = argparse.ArgumentParser()
parser.add_argument('model1', type=str)
parser.add_argument('infile1', type=str)
parser.add_argument('model2', type=str)
parser.add_argument('infile2', type=str)
args = parser.parse_args()

outfile = f'/bevfusion/{args.model1} vs {args.model2}.png'

dicts1 = []
with open(args.infile1, 'r') as f:
    lines = f.readlines()
    for line in lines:
        dicts1.append(json.loads(line))

dicts2 = []
with open(args.infile2, 'r') as f:
    lines = f.readlines()
    for line in lines:
        dicts2.append(json.loads(line))

epoch1 = []
mAP1 = []
nds1 = []
epoch2 = []
mAP2 = []
nds2 = []

iteration1 = []
loss1 = []
iteration2 = []
loss2 = []

for dict in dicts1:
    if 'mode' in dict:
        if dict['mode'] == 'val':
            epoch1.append(dict['epoch'])
            mAP1.append(dict['object/map'])
            nds1.append(dict['object/nds'])
        else:
            iteration1.append((dict['epoch'] - 1) * EPOCH_ITER + dict['iter'])
            loss1.append(dict['loss'])

for dict in dicts2:
    if 'mode' in dict:
        if dict['mode'] == 'val':
            epoch2.append(dict['epoch'])
            mAP2.append(dict['object/map'])
            nds2.append(dict['object/nds'])
        else:
            iteration2.append((dict['epoch'] - 1) * EPOCH_ITER + dict['iter'])
            loss2.append(dict['loss'])

length_eopch = min(len(epoch1), len(epoch2))
epoch1 = epoch1[:length_eopch]
mAP1 = mAP1[:length_eopch]
nds1 = nds1[:length_eopch]
epoch2 = epoch2[:length_eopch]
mAP2 = mAP2[:length_eopch]
nds2 = nds2[:length_eopch]

length_iter = min(len(iteration1), len(iteration2))
iteration1 = iteration1[:length_iter]
loss1 = loss1[:length_iter]
iteration2 = iteration2[:length_iter]
loss2 = loss2[:length_iter]

loss1 = np.array(loss1)
iteration1 = np.array(iteration1)[loss1 < 5]
loss1 = loss1[loss1 < 5]

loss2 = np.array(loss2)
iteration2 = np.array(iteration2)[loss2 < 5]
loss2 = loss2[loss2 < 5]

plt.figure(figsize=(max(10, len(epoch1), len(epoch2)), 15))
gs = gridspec.GridSpec(2, 2)

ax0 = plt.subplot(gs[0, :])
ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
ax0.plot(iteration1, loss1, label=args.model1, color='#1f77b4')
ax0.plot(iteration2, loss2, label=args.model2, color='#ff7f0e')
xmax = ax0.get_xlim()[1]
ymax = ax0.get_ylim()[1]
ax0.set_ylim(top=ymax * 1.05)
ax0.axvline(0, color='gray', linestyle='--', linewidth=1)
epoch_ends = list(range(EPOCH_ITER, max(iteration1[-1], iteration2[-1]) + LOG_INTERVAL, EPOCH_ITER))
for i, x in enumerate(epoch_ends):
    ax0.axvline(x, color='gray', linestyle='--', linewidth=1)
    ax0.text(x - EPOCH_ITER / 2, ymax, f'Epoch {i + 1}', ha='center', color='gray')
for x, y in zip(iteration1[::int(xmax // (LOG_INTERVAL * NUM_TEXT_LABELS))], loss1[::int(xmax // (LOG_INTERVAL * NUM_TEXT_LABELS))]):
    ax0.text(x, y + ymax * 0.03, f'{y:.3f}', ha='center', va='bottom', fontsize=10, color='#1f77b4')
for x, y in zip(iteration2[::int(xmax // (LOG_INTERVAL * NUM_TEXT_LABELS))], loss2[::int(xmax // (LOG_INTERVAL * NUM_TEXT_LABELS))]):
    ax0.text(x, y + ymax * 0.03, f'{y:.3f}', ha='center', va='bottom', fontsize=10, color='#ff7f0e')
ax0.set_xlabel('Iteration', fontsize=15)
ax0.tick_params(axis='both', labelsize=13)
ax0.set_ylabel('Loss', fontsize=15)
ax0.legend()

ax1 = plt.subplot(gs[1, 0])
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.plot(epoch1, mAP1, marker='o', label=args.model1, color='#1f77b4')
for x, y in zip(epoch1, mAP1):
    ax1.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10, color='#1f77b4')
ax1.plot(epoch2, mAP2, marker='o', label=args.model2, color='#ff7f0e')
for x, y in zip(epoch2, mAP2):
    ax1.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10, color='#ff7f0e')
ax1.hlines(ax1.get_yticks(), ax1.get_xlim()[0], ax1.get_xlim()[1], colors='gray', linestyles='--', linewidth=0.5)
ax1.set_xlabel('Epoch', fontsize=15)
ax1.tick_params(axis='both', labelsize=13)
ax1.set_ylabel('mAP', fontsize=15)
ax1.legend()

ax2 = plt.subplot(gs[1, 1])
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.plot(epoch1, nds1, marker='o', label=args.model1, color='#1f77b4')
for x, y in zip(epoch1, nds1):
    ax2.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10, color='#1f77b4')
ax2.plot(epoch2, nds2, marker='o', label=args.model2, color='#ff7f0e')
for x, y in zip(epoch2, nds2):
    ax2.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10, color='#ff7f0e')
ax2.hlines(ax2.get_yticks(), ax2.get_xlim()[0], ax2.get_xlim()[1], colors='gray', linestyles='--', linewidth=0.5)
ax2.set_xlabel('Epoch', fontsize=15)
ax2.tick_params(axis='both', labelsize=13)
ax2.set_ylabel('NDS', fontsize=15)
ax2.legend()

plt.tight_layout()
plt.savefig(outfile)
