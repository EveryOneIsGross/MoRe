import json
import matplotlib.pyplot as plt
import random

def plot_limited_charts(log_file, n_charts, random_selection=False):
    # Load the data
    with open(log_file, 'r') as f:
        log_data = json.load(f)

    # Limit the number of items to plot
    if random_selection:
        items_to_plot = random.sample(log_data['items'], min(n_charts, len(log_data['items'])))
    else:
        items_to_plot = log_data['items'][:n_charts]

    # Set up the plot style
    plt.figure(figsize=(15, 5 * len(items_to_plot)))

    # Create a subplot for each selected item
    for subplot_index, item in enumerate(items_to_plot, 1):
        plt.subplot(len(items_to_plot), 1, subplot_index)
        
        for comp_index, comparison in enumerate(item['comparisons'], 1):
            for model, model_data in comparison['models'].items():
                layers = [layer['layer'] for layer in model_data['layers']]
                scores = [layer['score'] for layer in model_data['layers']]
                plt.plot(layers, scores, marker='o', label=f"{model} (Comp {comp_index})")

        plt.xlabel('Layer')
        plt.ylabel('Score')
        plt.title(f'Item {item["item_index"]}: Scores Over Layers for Each Model and Comparison')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

    # Adjust the layout and display the plot
    plt.tight_layout()
    # Save the plot
    plt.savefig('mor_scores_visualization.png', bbox_inches='tight')
    print("\nVisualization saved as 'mor_scores_visualization.png'")
    plt.show()

    # Generate summary statistics
    print("\nSummary Statistics:")
    for item in items_to_plot:
        print(f"\nItem {item['item_index']}:")
        for comp_index, comparison in enumerate(item['comparisons'], 1):
            print(f"  Comparison {comp_index}:")
            for model, model_data in comparison['models'].items():
                scores = [layer['score'] for layer in model_data['layers']]
                print(f"    {model}:")
                print(f"      Min: {min(scores):.4f}")
                print(f"      Max: {max(scores):.4f}")
                print(f"      Mean: {sum(scores)/len(scores):.4f}")

# Usage
log_file = 'output_ranked_log.json'
n_charts = 3  # Set this to the number of charts you want to generate
random_selection = True  # Set to False if you want to select the first n items instead of random

plot_limited_charts(log_file, n_charts, random_selection)
