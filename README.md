# Conversation Recongition
This repository contains the work for the lecture Scene Understanding & Surveillance (SUS) 2022.
We detect if people have a conversation by checking if people are looking at each other (LAEO).

## Installation

The repository was tested with python version 3.8.13, torch 1.11 with cuda support.
Use following command to install the requirements:

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

## Usage

This repository contains scripts to predict, evaluate and check if people are looking at each other in the develop folder and real time inference with your webcam in the demo folder.

To make predictions use the predict.py script in the develop folder.
For example like this:
```
python develop/predict.py data/scenes/scene2.mp4 --prediction pred.txt --vispath test.mp4
```
where
 - ```--prediction``` path to store the prediction per frame
 - ```--vispath``` if given stores the visualization at this path
 - ```--vis``` if enabled visualizes the prediction during runtime

To calculate the accuracy, precision and recall as well as visualize them in a barplot you can run following command:
```
python develop/eval.py results predictions sota labels
```
where
 - *results* is the directory to store the results
 - *predictions* is the directory that contains all of the predicted videos
 - *sota* is the directory that contains the state-of-the-art results
 - *labels* is the directory that contains the ground truth labels

The predictions, sota and label files are expected to be named the same and contain a number and 1 for laeo and 0 for not laeo per line. Where one line corresponds to one frame in the video.


## Examples

...