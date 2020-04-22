import autokeras as ak
import pandas as pd
import numpy as np
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv("./data/test.csv")

train_test_data = [df_train, df_test]  # combining train and test dataset

# title_mapping = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs",
#                  "Master": "Boy", "Dr": "Gentleman", "Rev": "Gentleman", "Col": "unknown", "Major": "Gentleman", "Mlle": "unknown", "Countess": "Lady",
#                  "Ms": "Miss", "Lady": "Lady", "Jonkheer": "unknown", "Don": "Gentleman", "Dona": "Lady", "Mme": "unknown", "Capt": "Gentleman", "Sir": "Gentleman"}
# # features_drop = ['PassengerId', 'Ticket', 'Name', 'Title', 'Cabin', 'Family']
# features_drop = ['PassengerId', 'Ticket', 'Name', 'Family', 'Cabin']

# # create ageset
# concatset = (pd.concat([df_train, df_test]))
# concatset['Title'] = concatset['Name'].str.extract(
#     ' ([A-Za-z]+)\.', expand=False)
# # concatset['Title'] = concatset['Title'].map(title_mapping)
# ageset = concatset.groupby(['Title'])['Age'].mean()
# fareset = concatset.groupby(['Pclass'])['Fare'].mean()

# for dataset in train_test_data:
#     # mapping
#     dataset['Title'] = dataset['Name'].str.extract(
#         ' ([A-Za-z]+)\.', expand=False)
#     # concatset['Title'] = concatset['Title'].map(title_mapping)
#     dataset["Cabin"] = dataset["Cabin"].str[:1]

#     # fillna
#     # dataset["Age"].fillna(dataset.groupby(
#     #     "Title")["Age"].transform("median"), inplace=True)
#     dataset["Age"].fillna(dataset["Title"].map(ageset), inplace=True)
#     dataset['Embarked'] = dataset['Embarked'].fillna('unknown')
#     dataset['Cabin'] = dataset['Cabin'].fillna('unknown')
#     dataset["Fare"].fillna(dataset["Pclass"].map(fareset), inplace=True)
#     # dataset["Fare"].fillna(dataset.groupby("Pclass")[
#     #                        "Fare"].transform("median"), inplace=True)

#     # add
#     dataset["Family"] = dataset["SibSp"] + dataset["Parch"]
#     dataset['Alone'] = np.where(dataset['Family'] > 0, 'n', 'y')

#     # delete
#     dataset.drop(features_drop, axis=1, inplace=True)

# # save to csv
# df_train.to_csv("df_train.csv", mode='w')
# df_test.to_csv("df_test.csv", mode='w')


for dataset in train_test_data:
    dataset.drop('id', axis=1, inplace=True)

df_train.to_csv("df_train.csv", mode='w')
df_test.to_csv("df_test.csv", mode='w')

# # It tries n different models.
# clf = ak.StructuredDataClassifier(max_trials=100)
# # Feed the structured data classifier with training data.
# train_y = df_train.pop('price')
# clf.fit(x=df_train, y=train_y)

# It tries n different models.
clf = ak.StructuredDataRegressor()
# Feed the structured data classifier with training data.
train_y = df_train.pop('price')
clf.fit(x=df_train, y=train_y)

preds = clf.predict(df_test)
df_pred = pd.read_csv("./data/sample_submission.csv")
df_pred['price'] = preds
df_pred.to_csv(
    "./houseSubmission.csv", index=None)
