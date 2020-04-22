import autokeras as ak
import pandas as pd
df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv("./eval.csv")

# train_test_data = [df_train, df_test]  # combining train and test dataset

# sex_mapping = {"male": 0, "female": 1}
# title_mapping = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs",
#                  "Master": "Gentleman", "Dr": "Gentleman", "Rev": "Gentleman", "Col": "unknown", "Major": "Gentleman", "Mlle": "unknown", "Countess": "Lady",
#                  "Ms": "Miss", "Lady": "Lady", "Jonkheer": "unknown", "Don": "Gentleman", "Dona": "Lady", "Mme": "unknown", "Capt": "Gentleman", "Sir": "Gentleman"}
# cabin_mapping = {"A": 0, "B": 1, "C": 2,
#                  "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
# embarked_mapping = {"S": 0, "C": 1, "Q": 2}
# features_drop = ['PassengerId', 'Ticket', 'SibSp', 'Parch', 'Name']

# for dataset in train_test_data:
#     # mapping
#     dataset['Title'] = dataset['Name'].str.extract(
#         ' ([A-Za-z]+)\.', expand=False)
#     dataset['Title'] = dataset['Title'].map(title_mapping)
#     # dataset['Sex'] = dataset['Sex'].map(sex_mapping)
#     dataset["Cabin"] = dataset["Cabin"].str[:1]
#     dataset["Cabin"] = dataset["Cabin"].map(cabin_mapping)
#     # dataset["Embarked"] = dataset["Embarked"].map(embarked_mapping)

#     # fillna
#     dataset["Age"].fillna(dataset.groupby(
#         "Title")["Age"].transform("median"), inplace=True)
#     dataset['Embarked'] = dataset['Embarked'].fillna('S')
#     dataset["Fare"].fillna(dataset.groupby("Pclass")[
#                            "Fare"].transform("median"), inplace=True)
#     dataset["Cabin"].fillna(dataset.groupby(
#         "Pclass")["Cabin"].transform("median"), inplace=True)

#     # create
#     dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"]

#     # delete
#     dataset.drop(features_drop, axis=1, inplace=True)


# df_test.groupby("Title")["Age"].transform("median")
# # fill missing age with median age for each title (Mr, Mrs, Miss, Others)


# fill missing Fare with median fare for each Pclass
# df_train["Fare"].fillna(df_train.groupby(
#     "Pclass")["Fare"].transform("median"), inplace=True)
# df_test["Fare"].fillna(df_test.groupby("Pclass")[
#                        "Fare"].transform("median"), inplace=True)
# df_train.head(50)


# map to number
# cabin_mapping = {"A": 0, "B": 1, "C": 2,
#                  "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
# for dataset in train_test_data:
#     dataset["Cabin"] = dataset["Cabin"].str[:1]
#     dataset["Cabin"] = dataset["Cabin"].map(cabin_mapping)

# fill missing Fare with median fare for each Pclass
# df_train["Cabin"].fillna(df_train.groupby(
#     "Pclass")["Cabin"].transform("median"), inplace=True)
# df_test["Cabin"].fillna(df_test.groupby(
#     "Pclass")["Cabin"].transform("median"), inplace=True)


# df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"] + 1
# df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"] + 1

# # delete unnecessary feature from dataset
# features_drop = ['PassengerId', 'Ticket', 'SibSp', 'Parch', 'Name']
# df_train.drop(features_drop, axis=1, inplace=True)
# df_test.drop(features_drop, axis=1, inplace=True)
# df_train.head()

# Initialize the structured data classifier.
# It tries 1 different models.
clf = ak.StructuredDataClassifier(max_trials=100)
# Feed the structured data classifier with training data.
# train_y = df_train.pop('Survived')
clf.fit(x=df_train, y=df_train['survived'])
print('Accuracy: {accuracy}'.format(
    accuracy=clf.evaluate(x=df_test, y=df_test['survived'])))
# preds = clf.predict(df_test)
# df_pred = pd.read_csv("./titanic/gender_submission.csv")
# df_pred.head()
# df_pred['Survived'] = preds
# df_pred.head()
# df_pred.to_csv("./titanic_submission_feature2_try100.csv", index=None)
