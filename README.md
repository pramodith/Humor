# Baseline and Evaluation scripts for SemEval 2020 Task 7

The goal of this <a href="https://competitions.codalab.org/competitions/20970">shared task</a> is to assess humor in news headlines that have been modified using short edits to make them funny. There are two subtasks:

* Sub-task 1 (Funniness Estimation): Here the goal is to assign a funniness grade to an edited headline in the [0,3] interval. Systems will be ranked by Root Mean Squared Error.

* Sub-task 2 (Funnier of the Two): Given two differently edited versions of the same headline, the goal is to predict which is the funnier of the two. Systems will be ranked by prediction accuracy.

This repository provides python code to run baseline experiments and evaluation scripts for the two sub-tasks.

## Pre-requisites:
* Python 2 or 3

* Libraries: Pandas, Numpy

#### Sub-task 1:
Run the baseline (always predicts the overall mean funniness grade in the training set):

```
 python baseline_task_1.py ../data/task-1/train.csv ../data/task-1/dev.csv
```

#### Sub-task 2:
Run the baseline (always predicts the most-frequent label in the training set, i.e., headline 2):

```
python baseline_task_2.py ../data/task-2/train.csv ../data/task-2/dev.csv
```

## Model Trianing
The `Humor_Final.ipynb`notebook, is to reproduce the work of Pramodith pertaining to SemEval2020 Task 7. Assessing the Humor of Edited News Headlines. All required data files can be found in the git repo in which thie jupyter notebook is located. I know that the code hasn't been maintained properly since the work was done as an independent researcher in my spare time. If you have any doubts feel free to raise a GIT issue.

The notebook implements a Siamese Bert Architecture with a custom attention implementation.
