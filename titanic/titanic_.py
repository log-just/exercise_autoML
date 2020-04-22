import autokeras as ak
import pandas as pd
import numpy as np

# 테이터 읽기
df_train = pd.read_csv('./titanic/train.csv')
df_test = pd.read_csv("./titanic/test.csv")
train_test_data = [df_train, df_test]  # combining train and test dataset

# 전체 이름에서 매핑한 데이터
title_mapping = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs",
                 "Master": "Boy", "Dr": "Gentleman", "Rev": "Gentleman", "Col": "unknown", "Major": "Gentleman", "Mlle": "unknown", "Countess": "Lady",
                 "Ms": "Miss", "Lady": "Lady", "Jonkheer": "unknown", "Don": "Gentleman", "Dona": "Lady", "Mme": "unknown", "Capt": "Gentleman", "Sir": "Gentleman"}
# drop columns
features_drop = ['PassengerId', 'Ticket', 'Name', 'Family', 'Cabin']

# 훈련셋/테스트셋을 합쳐서.. Title별 평균 나이 / 등급별 평균 요금을 뽑는다 - 정확도를 높이기 위해 합친 셋으로 평균을 냄
concatSet = (pd.concat([df_train, df_test]))
concatSet['Title'] = concatSet['Name'].str.extract(
    ' ([A-Za-z]+)\.', expand=False)
ageSet = concatSet.groupby(['Title'])['Age'].mean()
fareSet = concatSet.groupby(['Pclass'])['Fare'].mean()

for dataset in train_test_data:
    # 천체이름을 Title로 매핑
    dataset['Title'] = dataset['Name'].str.extract(
        ' ([A-Za-z]+)\.', expand=False)
    # 선실은 첫 알파벳 숫자만 뽑는다
    dataset["Cabin"] = dataset["Cabin"].str[:1]

    # Nan 필드들 채우기
    dataset['Embarked'] = dataset['Embarked'].fillna('unknown')
    dataset['Cabin'] = dataset['Cabin'].fillna('unknown')
    # Nan 필드들 채우기 (나이/요금 평균)
    dataset["Age"].fillna(dataset["Title"].map(ageSet), inplace=True)
    dataset["Fare"].fillna(dataset["Pclass"].map(fareSet), inplace=True)

    # 혼자/가족 여부 컬럼 추가
    dataset["Family"] = dataset["SibSp"] + dataset["Parch"]
    dataset['Alone'] = np.where(dataset['Family'] > 0, 'n', 'y')

    # 삭제
    dataset.drop(features_drop, axis=1, inplace=True)

# 데이터 확인용으로 파일로 뽑기
df_train.to_csv("df_train.csv", mode='w')
df_test.to_csv("df_test.csv", mode='w')

# 모델 탐색 개수 100개로 훈련 (검증셋20% 할당)
clf = ak.StructuredDataClassifier(max_trials=100)
train_y = df_train.pop('Survived')
clf.fit(x=df_train, y=train_y, validation_split=0.2)

# 예측해서 파일로 뽑는다
preds = clf.predict(df_test)
df_pred = pd.read_csv("./titanic/gender_submission.csv")
df_pred['Survived'] = preds
df_pred.to_csv(
    "./titanic_submission_featureNoNan4_try100_val20.csv", index=None)
