import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Provided data
top_n_offender= {
            "0x7fa49602fc": {
                "num_correct_preds": 6364109,
                "num_incorrect_preds": 26491
            },
            "0x7fa4960250": {
                "num_correct_preds": 6381148,
                "num_incorrect_preds": 9452
            },
            "0x7fa4960290": {
                "num_correct_preds": 6382560,
                "num_incorrect_preds": 8040
            },
            "0x7fa496011c": {
                "num_correct_preds": 2605095,
                "num_incorrect_preds": 1305
            },
            "0x7fa496015c": {
                "num_correct_preds": 2605232,
                "num_incorrect_preds": 1168
            },
            "0x7fa4960170": {
                "num_correct_preds": 148236,
                "num_incorrect_preds": 564
            },
            "0x7fa4960164": {
                "num_correct_preds": 157880,
                "num_incorrect_preds": 520
            },
            "0x7fa4960114": {
                "num_correct_preds": 2576506,
                "num_incorrect_preds": 494
            },
            "0x7fa49600c0": {
                "num_correct_preds": 153114,
                "num_incorrect_preds": 486
            },
            "0x7fb7f1f25c": {
                "num_correct_preds": 2064,
                "num_incorrect_preds": 477
            },
            "0x7fb7f1f230": {
                "num_correct_preds": 2068,
                "num_incorrect_preds": 473
            },
            "0x7fb7f1e83c": {
                "num_correct_preds": 1778,
                "num_incorrect_preds": 330
            },
            "0x7fb7f1f254": {
                "num_correct_preds": 3138,
                "num_incorrect_preds": 326
            },
            "0x7fa49600dc": {
                "num_correct_preds": 153282,
                "num_incorrect_preds": 318
            },
            "0x7fa496006c": {
                "num_correct_preds": 1499691,
                "num_incorrect_preds": 309
            },
            "0x7fa496031c": {
                "num_correct_preds": 6390303,
                "num_incorrect_preds": 297
            }
        }

# Sorting the data by number of incorrect predictions in descending order
sorted_data = sorted(top_n_offender.items(), key=lambda x: x[1]['num_incorrect_preds'], reverse=True)

# Extract addresses and values for visualization
addresses = [item[0] for item in sorted_data]
num_incorrect = [item[1]['num_incorrect_preds'] for item in sorted_data]

# Calculate the proportion of incorrect predictions to total branch instructions
incorrect_ratio = [
    100*(item[1]['num_incorrect_preds'] / 59608)
    for item in sorted_data
]

# Set positions for side-by-side bars
bar_width = 0.8
index = np.arange(len(addresses))

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plotting incorrect predictions
hbars = ax1.barh(index, incorrect_ratio, bar_width, label='Incorrect Prediction Ratio', color='skyblue')
ax1.set_xlabel('Number of Incorrect Predictions')
ax1.set_ylabel('Addresses')
ax1.set_yticks(index)
ax1.set_yticklabels(addresses)
ax1.invert_yaxis()

# Add legend
ax1.legend(loc='upper right')
ax1.bar_label(hbars, fmt='%.2f')
# Display the plot
plt.title('Top n branch instructions MPKI %')
plt.tight_layout()

plt.savefig('test.png', dpi=300)
