import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    plt.title("Stacked Bar Chart with Optimized Label Positioning", fontsize=14, fontweight='bold')
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

    # Create a second y-axis for MPKI
    ax2 = ax1.twinx()
    color2 = "red"
    ax2.plot(df["br_inst_cnt"], df["mpki"], color=color2, label="MPKI")
    ax2.set_ylabel("MPKI", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add a title
    plt.title("Branch Prediction Accuracy and MPKI")

    # Save the plot as an image
    #output_image_path = "simulation_dual_axis_plot.png"  # Replace with your desired file name
    plt.savefig(output_image_path, dpi=300)
    