import pandas as pd

# Veri setini oku
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# İlk 5 satıra göz at
print(df.head())

print(df.info())

print(df.isnull().sum())

df.drop("Cabin", axis=1, inplace=True)
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
df["Sex"] = label.fit_transform(df["Sex"])  # male=1, female=0
df["Embarked"] = label.fit_transform(df["Embarked"])

df.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Doğruluk:", accuracy_score(y_test, y_pred))
print("\nKarışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
