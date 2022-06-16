


import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

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

        with open(f'{label_path}/{scene}', 'r') as f:
            labels = f.readlines()
            labels = np.array([int(l.split(' ')[1]) for l in labels])

        predictions = np.concatenate([predictions, np.zeros(len(labels)-len(predictions))])

        scene_name = scene[:-4]
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        results.append([scene_name, accuracy, precision, recall])
    return pd.DataFrame(results, columns=['scene', 'accuracy', 'precision', 'recall'])


if __name__ == '__main__':
    prediction_path = 'data/laeonet_pred'
    label_path = 'data/labels'
    laeo_net_df = calc_metrics(prediction_path, label_path)
    laeo_net_df['type'] = 'sota'
    prediction_path = 'data/predictions'
    pred_df = calc_metrics(prediction_path, label_path)
    pred_df['type'] = 'own'
    
    df = pd.concat([pred_df, laeo_net_df])
    df