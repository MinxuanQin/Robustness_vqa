import numpy as np
import matplotlib.pyplot as plt

# dump data
##### Antonym (attr)
# models = ['VisualBERT', 'ViLT', 'GIT', 'BLIP', 'OFA']
# metrics = ['ACC', 'C-ACC']
# original_scores = np.array([0.538, 0.722, 0.628, 0.513, 0.1])
# modified_scores = np.array([0.174, 0.565, 0.344, 0.200, 0.2])
#
# metrics = ['OrigACC', 'p_ACC', 'OrigWUPS', 'p_WUPS', 'ACC', 'C-ACC', 'WUPS', 'CONS']
# scores = [[0.294, 0.782, 0.442, 0.828, 0.538, 0.174, 0.635, 0.297],
#           [0.666, 0.779, 0.736, 0.825, 0.722, 0.565, 0.78, 0.624],
#           [0.562, 0.693, 0.655, 0.758, 0.628, 0.344, 0.706, 0.445],
#           [0.605, 0.421, 0.665, 0.523, 0.513, 0.200, 0.594, 0.384],
#           []]

##### Ontological (onto)
# models = ['VisualBERT', 'ViLT', 'GIT', 'BLIP', 'OFA']
# metrics = ['ACC', 'C-ACC']
# original_scores = np.array([0.458, 0.818, 0.777, 0.570, 0.1])
# modified_scores = np.array([0.236, 0.452, 0.427, 0.255, 0.2])
#
# metrics = ['OrigACC', 'p_ACC', 'OrigWUPS', 'p_WUPS', 'ACC', 'C-ACC', 'WUPS', 'CONS']
# scores = [[.409, .507, .555, .634, .458, .236, .595, .236],
#           [.826, .809, .877, .864, .818, .452, .871, .971],
#           [0.768, 0.786, 0.830, 0.845, 0.777, 0.427, 0.837, 0.979]
#           [0.610, 0.530, 0.700, 0.637, 0.570, 0.255, 0.669, 0.935],
#           []]

##### Phrasal (rephrase?)
# models = ['VisualBERT', 'ViLT', 'GIT', 'BLIP', 'OFA']
# metrics = ['ACC', 'C-ACC']
# original_scores = np.array([0.399,0.587,0.558,0.405,0.1])
# modified_scores = np.array([0.474,0.761,0.706,0.493,0.2])
#
# metrics = ['OrigACC', 'p_ACC', 'OrigWUPS', 'p_WUPS', 'ACC', 'C-ACC', 'WUPS', 'CONS']
# scores = [[0.400,0.397,0.613,0.612,0.399,0.474,0.612,0.952],
#           [0.588,0.587,0.734,0.733,0.587,0.761,0.734,0.915],
#           [0.558,0.558,0.710,0.711,0.558,0.706,0.711,0.912],
#           [0.406,0.404,0.542,0.540,0.405,0.493,0.541,0.872],
#           []]

##### Symmetry (order?)
# models = ['VisualBERT', 'ViLT', 'GIT', 'BLIP', 'OFA']
# metrics = ['ACC', 'C-ACC']
# original_scores = np.array([0.381,0.526,0.468,0.315,0.1])
# modified_scores = np.array([0.171,0.442,0.407,0.232,0.2])
#
# metrics = ['OrigACC', 'p_ACC', 'OrigWUPS', 'p_WUPS', 'ACC', 'C-ACC', 'WUPS', 'CONS']
# scores = [[0.378,0.384,0.630,0.632,0.381,0.171,0.631,0.575],
#           [0.526,0.527,0.708,0.709,0.526,0.442,0.708,0.836],
#           [0.467,0.468,0.657,0.658,0.468,0.407,0.657,0.820],
#           [0.316,0.314,0.464,0.464,0.315,0.232,0.464,0.668],
#           []]

##### Negation
# models = ['VisualBERT', 'ViLT', 'GIT', 'BLIP', 'OFA']
# metrics = ['ACC', 'C-ACC']
# original_scores = np.array([0.476,0.516,0.466,0.501,0.1])
# modified_scores = np.array([0.030,0.259,0.056,0.141,0.2])
#
# metrics = ['OrigACC', 'p_ACC', 'OrigWUPS', 'p_WUPS', 'ACC', 'C-ACC', 'WUPS', 'CONS']
# scores = [[0.458,0.494,0.566,0.600,0.476,0.030,0.583,0.050],
#           [0.632,0.400,0.698,0.526,0.516,0.259,0.612,0.324],
#           [0.675,0.257,0.734,0.409,0.466,0.056,0.571,0.082],
#           [0.532,0.471,0.625,0.574,0.501,0.141,0.599,0.203],
#           []]

##### baseline
models = ['VisualBERT', 'ViLT', 'GIT', 'BLIP', 'OFA']
metrics = ['ACC', 'WUPS', 'Score'] # ['ACC', 'C-ACC', 'WUPS', 'CONS']
scores = [[0.3272, 0.55596, 0.42015],
          [0.7229, 0.80941, 0.83223],
          [0.766, 0.82456, 0.85567],
          [0.699, 0.782, 0.781],
          [0.1, 0.1, 0.1]]
#

# Define colors for each model
colors = ['red', 'blue', 'green', 'orange', 'yellow']


# Define the data



# Plotting
fig, ax = plt.subplots()
bar_width = 0.15
opacity = 0.4
spacing = 0.15

for i, model in enumerate(models):
    x = np.arange(len(metrics)) + (bar_width * (i+1))
    ax.bar(x, scores[i], bar_width, alpha=opacity, color=colors[i], label=model)
    for j, score in enumerate(scores[i]):
        ax.text(x[j], score + 2, str(score), ha='center')

# ax.set_xlabel('Metrics')
# ax.set_ylabel('Scores')
ax.set_title('Evaluation Scores by Model and Metric')
ax.set_xticks(np.arange(len(metrics)) + (spacing * (len(models) / 2)))
ax.set_xticklabels(metrics)

# Remove tick lines for x-axis
ax.tick_params(axis='x', length=0)

# Adjust the plot layout to accommodate the legend
ax.set_ylim(top=np.max(scores) + 0.1)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_ylim(0, 1)

# Create the legend outside the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()

if __name__ == "__main__":
    main()