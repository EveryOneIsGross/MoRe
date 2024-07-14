import json
import matplotlib.pyplot as plt

# Load the data
with open('output_ranked_log.json', 'r') as f:
    log_data = json.load(f)

# Set up the plot style
plt.figure(figsize=(15, 5 * len(log_data['items'])))

# Create a subplot for each item
for item_index, item in enumerate(log_data['items'], 1):
    plt.subplot(len(log_data['items']), 1, item_index)
    
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
for item in log_data['items']:
    print(f"\nItem {item['item_index']}:")
    for comp_index, comparison in enumerate(item['comparisons'], 1):
        print(f"  Comparison {comp_index}:")
        for model, model_data in comparison['models'].items():
            scores = [layer['score'] for layer in model_data['layers']]
            print(f"    {model}:")
            print(f"      Min: {min(scores):.4f}")
            print(f"      Max: {max(scores):.4f}")
            print(f"      Mean: {sum(scores)/len(scores):.4f}")

