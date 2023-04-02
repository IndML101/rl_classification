import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve

cwd = os.path.dirname(os.path.abspath(__file__))
path_list = cwd.split('/')
project_folder_path = '/'.join(path_list[:-2])
sys.path.append(project_folder_path)

from rl_classification.environments.environments import TabularEnv
from rl_classification.algorithms.m2acla import M2ACLA


def feature_engineering(df, ct, train=True):

    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df['Cabin'] = df['Cabin'].isna().astype(int)

    if train:
        data = ct.fit_transform(df)
    else:
        data = ct.transform(df)

    data = pd.DataFrame(data, columns=ct.get_feature_names_out().tolist())
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    drop_cols = ['remainder__' + col for col in drop_cols]
    data.drop(drop_cols, axis=1, inplace=True)
    x = data.drop('remainder__Survived', axis=1).astype(np.float64)
    y = data['remainder__Survived'].astype(np.float64)
    
    return x, y


def genrate_classification_report(model,
                                  x_test,
                                  y_test,
                                  path=project_folder_path):
    ## predict y_hat
    environment = TabularEnv((x_test, y_test))
    y_hat = model.evaluate_value(environment)
    out_df = x_test.copy()
    out_df['remainder__Survived'] = y_test
    out_df['Pred_values'] = y_hat

    ## predict class
    fpr, tpr, thresholds = roc_curve(y_test, y_hat)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    print('\n')
    out_df['Predictions'] = (out_df['Pred_values'] > thresholds[ix]).map({
        True:
        1,
        False:
        0
    })

    # plot the roc curve for the model
    fig, ax = plt.subplots(figsize=[10, 7])
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Logistic')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(path, 'rl_classification/titanic/roc-curve.png'))

    ## print classification report
    print('*' * 15 + 'Classification Report' + '*' * 15)
    print('Accuracy score: {}'.format(
        accuracy_score(out_df['remainder__Survived'], out_df['Predictions'])))
    print('\n')
    print('Auc-Roc score: {}'.format(
        roc_auc_score(out_df['remainder__Survived'], out_df['Predictions'])))
    print('\n')
    print(classification_report(out_df['remainder__Survived'], out_df['Predictions']))
    print('\n')
    print(pd.crosstab(out_df['remainder__Survived'], out_df['Predictions']))

    return out_df


if __name__ == '__main__':

    train_data = pd.read_csv(
        os.path.join(project_folder_path,
                     'rl_classification/data/titanic/train.csv'))

    test_data = pd.read_csv(
        os.path.join(project_folder_path,
                     'rl_classification/data/titanic/test.csv'))

    train_data.dropna(inplace=True)
    # test_data.dropna(inplace=True)
    ct = ColumnTransformer(
        [('ohe', OneHotEncoder(drop='first'), ['Sex', 'Embarked'])],
        remainder='passthrough')
    x_train, y_train = feature_engineering(train_data, ct)
    # x_test, y_test = feature_engineering(test_data, ct, False)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=5, test_size=0.2)

    train_env = TabularEnv((x_train, y_train))

    m2acla = M2ACLA(train_env)
    actor, critic = m2acla.learn(timesteps=10000,
                                 model_path=os.path.join(
                                     project_folder_path,
                                     'rl_classification/models/titanic/'))

    out_df = genrate_classification_report(m2acla, x_test, y_test)

    fig, ax = plt.subplots(figsize=[10, 7])
    sns.boxplot(data=out_df, x='remainder__Survived', y='Pred_values', ax=ax)
    plt.savefig(
        os.path.join(project_folder_path,
                     'rl_classification/titanic/score-dist.png'))
