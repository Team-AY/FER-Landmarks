import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import pandas as pd

# TODO: add the ability to externally call with a folder name
with open('models/classifiers/relative_XY_Concat_20250105190815/eval_result.pkl', 'rb') as f:
   eval_result = pickle.load(f)
    
model_names = []
train_acc = []
test_acc = []

test_f1 = []
test_prec = []
test_recall = []

for (model_name, model_eval_result) in eval_result.items():
    model_names.append(model_name)
    train_acc.append(model_eval_result['accuracy']['train'])
    test_acc.append(model_eval_result['accuracy']['test'])

    test_f1.append(model_eval_result['f1']['test'])
    test_prec.append(model_eval_result['precision']['test'])
    test_recall.append(model_eval_result['recall']['test'])

def plot_acc():

    # Create a bar plot
    fig = plt.figure(figsize=(18, 6))  # Optional: Set figure size
    bar_width = 0.35  # Adjust bar width as needed
    index = range(len(model_names))  # Create x-axis positions for bars

    # Create bars for the first data set (data1)
    bars1 = plt.bar(index, train_acc, bar_width, label="Train Accuracy")

    # Create bars for the second data set (data2), shifted slightly on the x-axis
    bars2 = plt.bar([p + bar_width for p in index], test_acc, bar_width, label="Test Accuracy")

    # Set labels and title
    plt.xlabel("Model Names")
    plt.ylabel("Accuracy")
    plt.title("Train and Test Accuracy for Evaluated Models")

    # Add legend
    plt.legend()

    # Add values above each bar
    y_offset = 0  # Adjust y-offset for value placement above bars

    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar_width / 2, height + y_offset, f"{height:.3f}", ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar_width / 2, height + y_offset, f"{height:.3f}", ha='center', va='bottom')

    # Show the plot
    plt.xticks([p + bar_width / 2 for p in index], model_names)  # Adjust x-axis tick positions
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optional: Add gridlines

    plt.tight_layout()  # Adjust layout to prevent overlapping elements
    plt.show()

    fig.savefig(f"results/classifiers_accuracy.png")

def plot_f1_recall_prec():
    # Create a bar plot
    fig = plt.figure(figsize=(18, 6))  # Optional: Set figure size
    bar_width = 0.25  # Adjust bar width as needed
    index = range(len(model_names))  # Create x-axis positions for bars

    # Create bars for the first data set (data1)
    bars1 = plt.bar(index, test_f1, bar_width, label="F1")

    # Create bars for the second data set (data2), shifted slightly on the x-axis
    bars2 = plt.bar([p + bar_width for p in index], test_prec, bar_width, label="Precision")

    # Create bars for the third data set (data3), shifted slightly on the x-axis
    bars3 = plt.bar([p + 2*bar_width for p in index], test_recall, bar_width, label="Recall")

    # Set labels and title
    plt.xlabel("Model Names")
    plt.ylabel("Score")
    plt.title("Test Scores")

    # Add legend
    plt.legend()

    # Add values above each bar
    y_offset = 0  # Adjust y-offset for value placement above bars

    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar_width / 2, height + y_offset, f"{height:.3f}", ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar_width / 2, height + y_offset, f"{height:.3f}", ha='center', va='bottom')

    for bar in bars3:
        height = bar.get_height()
        plt.text(bar.get_x() + bar_width / 2, height + y_offset, f"{height:.3f}", ha='center', va='bottom')        

    # Show the plot
    plt.xticks([p + bar_width for p in index], model_names)  # Adjust x-axis tick positions
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optional: Add gridlines

    plt.tight_layout()  # Adjust layout to prevent overlapping elements
    plt.show()

    fig.savefig(f"results/classifiers__f1_recall_prec.png")

def plot_cm():
    # colormap of the confusion matrix
    style = sns.light_palette("green", as_cmap=True)

    # display all the columns and indexes
    pd.set_option('display.max_columns', None)

    # add the style and title

    # Create a figure with subplots
    #fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12, 6))  # Adjust figsize as needed    

    for (model_name, model_eval_result) in eval_result.items():
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))  # Adjust figsize as needed    
        ax_train = axes.flat[0]
        ax_test = axes.flat[1]

        sns.heatmap(model_eval_result['cm']['train'], annot=True, ax=ax_train, cmap=style, fmt='.3f', vmin=0, vmax=1)
        sns.heatmap(model_eval_result['cm']['test'], annot=True, ax=ax_test, cmap=style, fmt='.3f', vmin=0, vmax=1)        

        ax_train.set_title(f"{model_name} Confusion Matrix on Train Dataset", fontsize=12)
        ax_train.set_xlabel("Predicted Label", fontsize=10)
        ax_train.set_ylabel("True Label", fontsize=10)
        ax_train.tick_params(bottom=False, left=False, labelsize=10)  # Adjust tick labels and params

        ax_test.set_title(f"{model_name} Confusion Matrix on Test Dataset", fontsize=12)
        ax_test.set_xlabel("Predicted Label", fontsize=10)
        ax_test.set_ylabel("True Label", fontsize=10)
        ax_test.tick_params(bottom=False, left=False, labelsize=10)  # Adjust tick labels and params        

        fig.suptitle(f'{model_name}')
        fig.savefig(f"results/{model_name}_cm.png")
        #model_eval_result['cm']['train'].style.set_caption(f"<h4 style='text-align: center'>{model_name} Confusion Matrix on Train Dataset</h1>").background_gradient(cmap=style)    
        #model_eval_result['cm']['train'].plot(subplots=True, ax=[counter, counter+6])
        #plt.subplot(6,2,counter+1)
        #model_eval_result['cm']['test'].style.set_caption(f"<h4 style='text-align: center'>{model_name} Confusion Matrix on Test Dataset</h1>").background_gradient(cmap=style)          
    plt.show()

plot_acc()
plot_f1_recall_prec()
plot_cm()