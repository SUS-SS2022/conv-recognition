import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt

def calc_metrics(prediction_path, label_path):

    results = []

    scenes = os.listdir(prediction_path)
    scenes.sort()

    for scene in scenes:    
        pred_exist = False

        with open(f'{prediction_path}/{scene}', 'r') as f:
            predictions = f.readlines()
            predictions = np.array([int(p.split(' ')[1]) for p in predictions])
            pred_exist = True

        if not pred_exist:
            continue
        
        label_exist = False
        with open(f'{label_path}/{scene}', 'r') as f:
            labels = f.readlines()
            labels = np.array([int(l.split(' ')[1]) for l in labels])
            label_exist = True

        if not label_exist:
            continue

        predictions = np.concatenate([predictions, np.zeros(len(labels)-len(predictions))])

        scene_name = scene[:-4]
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        results.append([scene_name, accuracy, precision, recall])

    return pd.DataFrame(results, columns=['scene', 'accuracy', 'precision', 'recall'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label images with laeo detector')
    parser.add_argument('path',
                        help='Path to save the evaluation plots and result csv')
    parser.add_argument('prediction',
                        help='path to conv-recognition predictions')
    parser.add_argument('sota',
                        help='path to state of the art predictions')
    parser.add_argument('labels',
                        help='path to labels')

    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f'{args.path} not a directory')
        exit()

    prediction_path = args.prediction
    label_path = args.labels
    sota_path = args.sota

    laeo_net_df = calc_metrics(prediction_path, label_path)
    laeo_net_df['type'] = 'sota'

    pred_df = calc_metrics(sota_path, label_path)
    pred_df['type'] = 'own'

    df = pd.concat([pred_df, laeo_net_df])
    df.to_csv(f'{args.path}/eval.csv', index=False)

    fig = sns.catplot(data=df, x='type', y='accuracy', col='scene', kind='bar', height=4, aspect=.7)
    plt.show()
    fig.savefig(f'{args.path}/accuracy.pdf')

    fig = sns.catplot(data=df, x='type', y='precision', col='scene', kind='bar', height=4, aspect=.7)
    plt.show()
    fig.savefig(f'{args.path}/precision.pdf')

    fig = sns.catplot(data=df, x='type', y='recall', col='scene', kind='bar', height=4, aspect=.7)
    plt.show()
    fig.savefig(f'{args.path}/recall.pdf')