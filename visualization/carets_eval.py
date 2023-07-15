import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# dump data

##### Antonym (attr)
class Antonym:
    name = "Antonym"
    models = ['VisualBERT', 'ViLT', 'GIT', 'BLIP', 'OFA', 'Human']
    metrics = ['ACC', 'C-ACC']
    original_scores = np.array([0.538, 0.722, 0.628, 0.513, 0.1, 0.875])
    modified_scores = np.array([0.174, 0.565, 0.344, 0.200, 0.2, 0.840])

    metrics = ['OrigACC', 'p_ACC', 'OrigWUPS', 'p_WUPS', 'ACC', 'C-ACC', 'WUPS', 'CONS']
    scores = np.array([[0.294, 0.782, 0.442, 0.828, 0.538, 0.174, 0.635, 0.297],
              [0.666, 0.779, 0.736, 0.825, 0.722, 0.565, 0.78, 0.624],
              [0.562, 0.693, 0.655, 0.758, 0.628, 0.344, 0.706, 0.445],
              [0.605, 0.421, 0.665, 0.523, 0.513, 0.200, 0.594, 0.384],
              [.1, .1, .1, .1, .1, .1, .1, .1]])

    def __init__(self):
        pass

##### Ontological (onto)
class Ontological:
    name = "Ontological"
    models = ['VisualBERT', 'ViLT', 'GIT', 'BLIP', 'OFA', 'Human']
    metrics = ['ACC', 'C-ACC']
    original_scores = np.array([0.458, 0.818, 0.777, 0.570, 0.1, 0.96])
    modified_scores = np.array([0.236, 0.452, 0.427, 0.255, 0.2, 0.94])

    metrics = ['OrigACC', 'p_ACC', 'OrigWUPS', 'p_WUPS', 'ACC', 'C-ACC', 'WUPS', 'CONS']
    scores = np.array([[.409, .507, .555, .634, .458, .236, .595, .236],
              [.826, .809, .877, .864, .818, .452, .871, .971],
              [0.768, 0.786, 0.830, 0.845, 0.777, 0.427, 0.837, 0.979],
              [0.610, 0.530, 0.700, 0.637, 0.570, 0.255, 0.669, 0.935],
              [.1, .1, .1, .1, .1, .1, .1, .1]])

    def __init__(self):
        pass

##### Phrasal (rephrase?)
class Phrasal:
    name = "Phrasal"
    models = ['VisualBERT', 'ViLT', 'GIT', 'BLIP', 'OFA', 'Human']
    metrics = ['ACC', 'C-ACC']
    original_scores = np.array([0.399,0.587,0.558,0.405,0.1,0.862])
    modified_scores = np.array([0.474,0.761,0.706,0.493,0.2,0.853])

    metrics = ['OrigACC', 'p_ACC', 'OrigWUPS', 'p_WUPS', 'ACC', 'C-ACC', 'WUPS', 'CONS']
    scores = np.array([[0.400,0.397,0.613,0.612,0.399,0.474,0.612,0.952],
              [0.588,0.587,0.734,0.733,0.587,0.761,0.734,0.915],
              [0.558,0.558,0.710,0.711,0.558,0.706,0.711,0.912],
              [0.406,0.404,0.542,0.540,0.405,0.493,0.541,0.872],
              [.1, .1, .1, .1, .1, .1, .1, .1]])

    def __init__(self):
        pass

##### Symmetry (order?)
class Symmetry:
    name = "Symmetry"
    models = ['VisualBERT', 'ViLT', 'GIT', 'BLIP', 'OFA', 'Human']
    metrics = ['ACC', 'C-ACC']
    original_scores = np.array([0.381,0.526,0.468,0.315,0.1,0.791])
    modified_scores = np.array([0.171,0.442,0.407,0.232,0.2,0.788])

    metrics = ['OrigACC', 'p_ACC', 'OrigWUPS', 'p_WUPS', 'ACC', 'C-ACC', 'WUPS', 'CONS']
    scores = np.array([[0.378,0.384,0.630,0.632,0.381,0.171,0.631,0.575],
              [0.526,0.527,0.708,0.709,0.526,0.442,0.708,0.836],
              [0.467,0.468,0.657,0.658,0.468,0.407,0.657,0.820],
              [0.316,0.314,0.464,0.464,0.315,0.232,0.464,0.668],
              [.1, .1, .1, .1, .1, .1, .1, .1]])

    def __init__(self):
        pass


##### Negation
class Negation:
    name = "Negation"
    models = ['VisualBERT', 'ViLT', 'GIT', 'BLIP', 'OFA', 'Human']
    metrics = ['ACC', 'C-ACC']
    original_scores = np.array([0.476,0.516,0.466,0.501,0.1,0.782])
    modified_scores = np.array([0.030,0.259,0.056,0.141,0.2,0.745])

    metrics = ['OrigACC', 'p_ACC', 'OrigWUPS', 'p_WUPS', 'ACC', 'C-ACC', 'WUPS', 'CONS']
    scores = np.array([[0.458,0.494,0.566,0.600,0.476,0.030,0.583,0.050],
              [0.632,0.400,0.698,0.526,0.516,0.259,0.612,0.324],
              [0.675,0.257,0.734,0.409,0.466,0.056,0.571,0.082],
              [0.532,0.471,0.625,0.574,0.501,0.141,0.599,0.203],
              [.1, .1, .1, .1, .1, .1, .1, .1]])

    def __init__(self):
        pass

#####visual?

def main():
    for test in [Antonym(), Ontological(), Symmetry(), Phrasal(), Negation()]:
        # Define the data
        name = test.name
        models = test.models
        metrics = test.metrics
        original_scores = test.original_scores
        modified_scores = test.modified_scores

        # Define colors for each model
        colors = ['red', 'blue', 'green', 'orange', 'yellow', 'gray']
        assert len(colors) == len(models)

        # Plotting
        fig, ax = plt.subplots()
        bar_width = 0.8
        opacity = 0.3
        handles = []  # Keep track of legend handles

        for i, model in enumerate(models):
            x = i + bar_width
            diff = original_scores[i] - modified_scores[i]
            rect = ax.bar(x, original_scores[i], width=bar_width, alpha=opacity, color=colors[i], label=model, align='center')
            handles.extend(rect)

            degrade = False
            if diff > 0: # if performance degrades
                start = original_scores[i] - diff
                degrade = True
            else:
                start = original_scores[i]

            r = ax.bar(x, abs(diff), bar_width, bottom=start,
                       alpha=opacity + 0.3, color=colors[i], label=model,align='center')
            props=dict(headwidth=20, headlength=1 if abs(diff) < 0.03 else 8, width=8, color="black", alpha=opacity)
            plt.annotate("", xy=(rect[0].get_x() + rect[0].get_width() / 2., original_scores[i] - diff),
                         xytext=(rect[0].get_x() + rect[0].get_width() / 2., original_scores[i]),
                         arrowprops=props)
            fig.texts.append(ax.texts.pop()) # do not ignore alpha

            # annotate values
            padding = 0 if degrade else -14 # score displayed top/bottom of the original bar
            ax.bar_label(rect, padding=padding)
            height = rect[0].get_height() - diff - 0.04 if degrade else rect[0].get_height() - diff
            ax.text(
                r[0].get_x() + r[0].get_width() / 2, height, modified_scores[i], ha="center", va="bottom"
            )

        # ax.set_xlabel('Metrics')
        # ax.set_ylabel('Performance Difference')
        # ax.set_xticks([0.5])#np.arange(len(metrics)) + (spacing * (len(models) / 2)))
        # ax.set_xticklabels(metrics)
        ax.set_title(name)#('Performance Difference by Model and Metric')
        ax.set_xlabel(r"ACC $\rightarrow$ C-ACC")
        ax.set_ylim(0, 1)

        # Remove tick lines for x-axis
        ax.tick_params(axis='x', length=0)
        ax.set_xticks([])

        # Create the legend outside the plot
        ax.legend(handles=handles, labels=models, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    main()