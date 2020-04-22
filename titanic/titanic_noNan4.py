import autokeras as ak
import pandas as pd
import numpy as np

# load file
df_train = pd.read_csv('./titanic/train.csv')
df_test = pd.read_csv("./titanic/test.csv")
train_test_data = [df_train, df_test]  # combining train and test dataset

# mapping Data
title_mapping = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs",
                 "Master": "Boy", "Dr": "Gentleman", "Rev": "Gentleman", "Col": "unknown", "Major": "Gentleman", "Mlle": "unknown", "Countess": "Lady",
                 "Ms": "Miss", "Lady": "Lady", "Jonkheer": "unknown", "Don": "Gentleman", "Dona": "Lady", "Mme": "unknown", "Capt": "Gentleman", "Sir": "Gentleman"}

features_drop = ['PassengerId', 'Ticket', 'Name', 'Family', 'Cabin']

# ageset/fare
concatSet = (pd.concat([df_train, df_test]))
concatSet['Title'] = concatSet['Name'].str.extract(
    ' ([A-Za-z]+)\.', expand=False)
# concatset['Title'] = concatset['Title'].map(title_mapping)
ageSet = concatSet.groupby(['Title'])['Age'].mean()
fareSet = concatSet.groupby(['Pclass'])['Fare'].mean()

for dataset in train_test_data:
    # mapping
    dataset['Title'] = dataset['Name'].str.extract(
        ' ([A-Za-z]+)\.', expand=False)
    # concatset['Title'] = concatset['Title'].map(title_mapping)
    dataset["Cabin"] = dataset["Cabin"].str[:1]

    # fillna
    # dataset["Age"].fillna(dataset.groupby(
    #     "Title")["Age"].transform("median"), inplace=True)
    dataset["Age"].fillna(dataset["Title"].map(ageSet), inplace=True)
    dataset['Embarked'] = dataset['Embarked'].fillna('unknown')
    dataset['Cabin'] = dataset['Cabin'].fillna('unknown')
    dataset["Fare"].fillna(dataset["Pclass"].map(fareSet), inplace=True)
    # dataset["Fare"].fillna(dataset.groupby("Pclass")[
    #                        "Fare"].transform("median"), inplace=True)

    # add
    dataset["Family"] = dataset["SibSp"] + dataset["Parch"]
    dataset['Alone'] = np.where(dataset['Family'] > 0, 'n', 'y')

    # delete
    dataset.drop(features_drop, axis=1, inplace=True)

# save to csv
df_train.to_csv("df_train.csv", mode='w')
df_test.to_csv("df_test.csv", mode='w')

# It tries n different models.
clf = ak.StructuredDataClassifier(max_trials=100)
# Feed the structured data classifier with training data.
train_y = df_train.pop('Survived')
clf.fit(x=df_train, y=train_y, validation_split=0.3)
preds = clf.predict(df_test)
df_pred = pd.read_csv("./titanic/gender_submission.csv")
df_pred['Survived'] = preds
df_pred.to_csv(
    "./titanic_submission_featureNoNan4_try100_val30.csv", index=None)
