import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from . import helpers

def plot_per_class(df, output_image_path):
    stats = df.groupby('class').agg({'num_incorrect_preds': ['median', 'mean', 'sum', 'count', 'std']})

    stats.columns = ['_'.join(col).strip() for col in stats.columns]
    #print(stats)

    fig = plt.figure(figsize=(14,10))
    gs = fig.add_gridspec(
        nrows = 2, ncols= 2
    )
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:])
    #fig, (ax1, ax3) = plt.subplots(2,2,figsize=(14,6), sharey=False)
    fig.suptitle('Branch class statistics', fontsize = 16)

    x = np.arange(len(stats.index))
    #TODO maybe move out decode class to some other module (i.e tools.py)
    helper = helpers.helper()
    xticks = [helper.decode_class(i) for i in stats.index]
    w = 0.35

    meanbar = ax1.bar(x - w/2, stats['num_incorrect_preds_mean'], w, label='Mean', color = 'skyblue', yerr = stats['num_incorrect_preds_std'], capsize=10)
    medbar = ax1.bar(x + w/2, stats['num_incorrect_preds_median'], w, label='Median', color = 'salmon')
    #sumbar = ax1.bar(x + w/3, stats['num_incorrect_preds_sum'], w, label = 'Sum', color = 'lightgreen')

    ax1.set_title('Mean/median per branch class (log)')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Mispredictions')
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(xticks, fontsize=7, rotation=40)
    ax1.legend()
    ax1.grid(axis = 'y', linestyle='--', alpha=0.7)

    for bar in meanbar + medbar:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')

    counts = stats['num_incorrect_preds_count']
    explode = [0.1 if (v / sum(counts)) < 0.05 else 0 for v in counts]
    ax2.pie(stats['num_incorrect_preds_count'], labels=xticks, 
            autopct='%1.1f%%', explode=explode,
            rotatelabels=True, 
            textprops={'fontsize': 7})
    ax2.set_title('Class frequency')

    sumbar = ax3.bar(x, stats['num_incorrect_preds_sum'], w, color = 'lightgreen')
    ax3.set_title('Total mispredictions (log)')
    ax3.set_xlabel('Class')
    ax3.set_xticks(x)
    ax3.set_xticklabels(xticks, fontsize=8)
    ax3.set_ylabel('Total mispredictions')
    ax3.set_yscale('log')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in sumbar:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height}', ha='center', va='bottom')

    plt.tight_layout()

    plt.savefig(output_image_path, dpi=300)

def plot_top_n_sum(top_n_addr, total_mispred_cnt, output_image_path):
    num  = [i for i in range(1, len(top_n_addr) + 1)]
    num_incorrect = [top_n_addr[item]['num_incorrect_preds'] for item in top_n_addr]

    incorrect_ratio = [
        100*(sum(num_incorrect[:i]) / total_mispred_cnt)
        for i in range(1, len(num_incorrect) + 1)
    ]
    # Set positions for side-by-side bars
    bar_width = 0.8
    index = np.arange(len(num))

    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plotting incorrect predictions
    hbars = ax1.barh(index, incorrect_ratio, bar_width, label='Incorrect Prediction Ratio', color='skyblue')
    ax1.set_xlabel('% of MPKI')
    ax1.set_ylabel('Top N offender')
    ax1.set_yticks(index)
    ax1.set_yticklabels(num)
    ax1.invert_yaxis()

    # Add legend
    ax1.legend(loc='upper right')
    ax1.bar_label(hbars, fmt='%.2f')
    # Display the plot
    plt.title('Top n branch instructions MPKI %')
    plt.tight_layout()

    plt.savefig(output_image_path, dpi=300)

def plot_top_n_addr(top_n_addr, total_mispred_cnt, output_image_path):
    addrs  = [hex(int(item)) for item in top_n_addr]
    num_incorrect = [top_n_addr[item]['num_incorrect_preds'] for item in top_n_addr]

    incorrect_ratio = [
        100*(item / total_mispred_cnt)
        for item in num_incorrect
    ]
    # Set positions for side-by-side bars
    bar_width = 0.8
    index = np.arange(len(addrs))

    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plotting incorrect predictions
    hbars = ax1.barh(index, incorrect_ratio, bar_width, label='Incorrect Prediction Ratio', color='skyblue')
    ax1.set_xlabel('% of MPKI')
    ax1.set_ylabel('Addresses')
    ax1.set_yticks(index)
    ax1.set_yticklabels(addrs)
    ax1.invert_yaxis()

    # Add legend
    ax1.legend(loc='upper right')
    ax1.bar_label(hbars, fmt='%.2f')
    # Display the plot
    plt.title('Top n branch instructions MPKI % per address')
    plt.tight_layout()

    plt.savefig(output_image_path, dpi=300)


def plot_storage_bar(storage_report, output_image_path, logger):
    key2label = {
        'ghist_size_b': 'Global history reg',
        'phist_size_b': 'Path history reg',
        'use_alt_on_new_alloc': 'misc',
        "pred_bit_b": 'Base pred bits',
        "pred_hyst_b": 'Base hyst bits',
    }
    # create Kb version of the stats
    data = {}
    for k, v in storage_report.items():
        if k == 'base':
            for k1,v1 in v.items():
                label = key2label[k1]
                data[label] = np.float64(v1 / 1024)
        elif k == 'tagged':
            for k1,v1 in v.items():
                label = f'Tagged predictor {k1[:-2]}'
                data[label] = np.float64(v1 / 1024)
        elif k == 'tot_size_b':
            continue
        else:
            label = key2label[k]
            data[label] = np.float64(v / 1024)
    logger.info(f'storage in kb: {data}')
    #print(data)

    labels = list(data.keys())
    values = list(data.values())

    # Create figure and plot stacked bar chart
    fig, ax = plt.subplots(figsize=(18, 8))

    # Center the graph in the figure
    ax.set_position([0.1, 0.2, 0.8, 0.6])  # [left, bottom, width, height]

    y_pos = 0
    left_edge = 0
    colors = plt.cm.tab20.colors  # Color variety

    for i, (label, val) in enumerate(zip(labels, values)):
        ax.barh(
            y_pos, width=val, left=left_edge,
            color=colors[i % len(colors)], edgecolor='black'
        )
        left_edge += val

    ax.set_yticks([])  # Hide Y-axis ticks
    ax.set_xlabel("Storage (kb)", fontsize=12)
    total_storage = sum(values)
    ax.set_xlim(0, total_storage * 1.1)  # Extra space on the right for readability

    # Extend arrows outside the bar with better label positioning
    arrowprops = dict(arrowstyle="->", color='black', linewidth=1.2)
    font_size = 10

    cumulative_left = 0
    y_top_offset = 1.4  # Higher offset to avoid clutter
    y_bottom_offset = -1.4  # Lower offset for alternating labels

    for i, (label, val) in enumerate(zip(labels, values)):
        center_x = cumulative_left + val / 2
        center_y = y_pos  # Bar position

        # Alternate labels above and below
        label_y = y_top_offset if i % 2 == 0 else y_bottom_offset

        # Handle small slices with rotation
        rotation = 0
        ha = 'center'
        if val < 5:
            if label_y == y_top_offset:
                rotation = 45
                ha = 'right'
            else:
                rotation = -45
                ha = 'right'

        ax.annotate(
            label, 
            xy=(center_x, center_y),
            xytext=(center_x, center_y + label_y),
            arrowprops=arrowprops,
            ha=ha, va='bottom',
            fontsize=font_size,
            rotation=rotation
        )

        cumulative_left += val

    # Add horizontal total width indicator
    ax.annotate(
        f"Total: {total_storage:.2f} kB",
        xy=(total_storage, y_pos),
        xytext=(total_storage, y_pos - 1.8),  # Position below the bar
        fontsize=12, fontweight='bold', color='red',
        ha='center', va='center',
        arrowprops=dict(arrowstyle='-', color='red', linewidth=1.5)
    )

    # Adjust plot limits to prevent text from going off screen
    ax.set_ylim(-3, 3)  # Extra space above and below the bar

    # Final title styling
    plt.title("Storage budget bar chart", fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    #plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)

def plot_mpki_accuracy(df, output_image_path):
    # Create the figure and axis objects
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Accuracy on the left y-axis
    color1 = "blue"
    ax1.plot(df["br_inst_cnt"], df["accuracy"], color=color1, label="Accuracy")
    ax1.set_xlabel("Branch Instruction Count")
    ax1.set_ylabel("Accuracy", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.set_xscale('log')
    asymptote_acc = df["accuracy"].iloc[-len(df)//10:].mean()
    ax1.axhline(asymptote_acc, color = color1, linestyle='--', alpha=0.7, label=f'{asymptote_acc:.2f}')
    ax1.annotate(f'{asymptote_acc:.4f}', 
                xy=(1.02, asymptote_acc),
                xycoords=('axes fraction', 'data'),
                color=color1,
                va='center',
                ha='left',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1),
                )


    # Create a second y-axis for MPKI
    ax2 = ax1.twinx()
    color2 = "red"
    ax2.plot(df["br_inst_cnt"], df["mpki"], color=color2, label="MPKI")
    ax2.set_ylabel("MPKI", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_xscale('log')
    asymptote_mpki = df["mpki"].iloc[-len(df)//10:].mean()
    ax2.axhline(asymptote_mpki, color = color2, 
                linestyle='--', alpha=0.7, 
                label=f'{asymptote_mpki:.2f}')

    ax2.annotate(f'{asymptote_mpki:.4f}', 
                xy=(1.02, asymptote_mpki),
                xycoords=('axes fraction', 'data'),
                color=color2,
                va='center',
                ha='left',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1),
                )

    # Add a title
    plt.title("Branch Prediction Accuracy and MPKI")

    # Save the plot as an image
    plt.savefig(output_image_path, dpi=300)
    
