import pickle 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def load_labeled_data(file_path):
    with open(file_path, 'rb') as file:
        labeled_data = pickle.load(file)
    return labeled_data
loaded_labeled_data = load_labeled_data("./model/model3/labeled_data.pkl")

X = [features for features, label in loaded_labeled_data]
y = [label for features, label in loaded_labeled_data]



if(True):
    exit

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)


# model evaluating
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)


def save_labeled_data(labeled_data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(labeled_data, file)


file_path = "C:\\Users\\Gen3r\\Documents\\capstone\\ml_model\\model\\model3\\gesture_recognition_data.pkl"
save_labeled_data(model, file_path)