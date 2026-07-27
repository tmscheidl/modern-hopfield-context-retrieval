import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ============================================================
# Parse training log files
# ============================================================

def parse_log(filepath):
    epochs = []
    train_loss = []
    val_loss = []
    dauprc_val = []

    with open(filepath, 'r') as f:
        content = f.read()

    # Each epoch appears twice in the log (start and end)
    # We want the final value for each epoch (second occurrence)
    # Pattern: "Epoch N: 100%|..." with metrics
    pattern = r'Epoch (\d+): 100%.*?val_loss=([\d.]+), dAUPRC_val=([\d.]+).*?train_loss=([\d.]+)'
    matches = re.findall(pattern, content)

    # Keep only last occurrence of each epoch
    epoch_data = {}
    for match in matches:
        epoch = int(match[0])
        epoch_data[epoch] = {
            'val_loss': float(match[1]),
            'dauprc_val': float(match[2]),
            'train_loss': float(match[3]),
        }

    for epoch in sorted(epoch_data.keys()):
        epochs.append(epoch)
        val_loss.append(epoch_data[epoch]['val_loss'])
        dauprc_val.append(epoch_data[epoch]['dauprc_val'])
        train_loss.append(epoch_data[epoch]['train_loss'])

    return epochs, train_loss, val_loss, dauprc_val


# ============================================================
# Load data
# ============================================================

runs = {
    'V1 (Cross→Context, lr=1e-4)': 'training_output.txt',
    'V2 (Context→Cross, lr=1e-4)': 'training_output_v2.txt',
    'V3 (Cross→Context, lr=1e-5, dropout=0.3)': 'training_output_v3.txt',
}

colors = {
    'V1 (Cross→Context, lr=1e-4)': '#2196F3',
    'V2 (Context→Cross, lr=1e-4)': '#FF9800',
    'V3 (Cross→Context, lr=1e-5, dropout=0.3)': '#4CAF50',
}

parsed = {}
for name, filepath in runs.items():
    try:
        epochs, train_loss, val_loss, dauprc_val = parse_log(filepath)
        parsed[name] = {
            'epochs': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'dauprc_val': dauprc_val,
        }
        print(f"Loaded {name}: {len(epochs)} epochs, best dAUPRC_val={max(dauprc_val):.4f} at epoch {epochs[dauprc_val.index(max(dauprc_val))]}")
    except FileNotFoundError:
        print(f"Warning: {filepath} not found, skipping {name}")

# ============================================================
# Plot
# ============================================================

fig = plt.figure(figsize=(16, 5))
fig.patch.set_facecolor('#1a1a2e')

gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

axes = [ax1, ax2, ax3]
titles = ['Training Loss', 'Validation Loss', 'dAUPRC Validation']
keys = ['train_loss', 'val_loss', 'dauprc_val']

for ax, title, key in zip(axes, titles, keys):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', color='#aaaaaa', fontsize=10)
    ax.tick_params(colors='#aaaaaa', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.grid(True, color='#2a2a4a', linewidth=0.5, linestyle='--')

    for name, data in parsed.items():
        short_name = name.split(' ')[0]  # V1, V2, V3
        ax.plot(
            data['epochs'],
            data[key],
            color=colors[name],
            linewidth=2,
            label=short_name,
            alpha=0.9,
        )

        # Mark best epoch for dAUPRC_val
        if key == 'dauprc_val':
            best_idx = data[key].index(max(data[key]))
            ax.axvline(
                x=data['epochs'][best_idx],
                color=colors[name],
                linewidth=1,
                linestyle=':',
                alpha=0.5,
            )
            ax.scatter(
                [data['epochs'][best_idx]],
                [data[key][best_idx]],
                color=colors[name],
                s=60,
                zorder=5,
            )

    ax.legend(
        facecolor='#1a1a2e',
        edgecolor='#444466',
        labelcolor='white',
        fontsize=9,
    )

# Add summary annotation
best_overall = {}
for name, data in parsed.items():
    best_val = max(data['dauprc_val'])
    best_ep = data['epochs'][data['dauprc_val'].index(best_val)]
    best_overall[name] = (best_val, best_ep)

summary_lines = ['Best dAUPRC_val:']
for name, (val, ep) in best_overall.items():
    short = name.split(' ')[0]
    summary_lines.append(f'  {short}: {val:.4f} @ epoch {ep}')

fig.text(
    0.5, -0.05,
    '  |  '.join(summary_lines),
    ha='center', va='top',
    color='#aaaaaa', fontsize=9,
    transform=fig.transFigure,
)

fig.suptitle(
    'MHNfs Training Curves — V1 / V2 / V3 Comparison',
    color='white', fontsize=14, fontweight='bold', y=1.02,
)

plt.savefig(
    'training_curves.png',
    dpi=150,
    bbox_inches='tight',
    facecolor='#1a1a2e',
)

print("\nSaved: training_curves.png")