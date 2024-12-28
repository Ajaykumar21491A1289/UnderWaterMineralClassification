from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import pickle

app = Flask(__name__)

# Load your dataset
main_df = pd.read_csv('sonar.csv', header=None)
X = main_df.drop(columns=[60])
y = main_df[60].apply(lambda x: 1 if x == 'M' else 0)

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define resampling strategy
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train AdaBoost classifier
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
ada_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, algorithm='SAMME', random_state=42)
ada_boost.fit(X_train_resampled, y_train_resampled)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(ada_boost, file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided in the request.'}), 400

        features = data['features']

        if not isinstance(features, list) or len(features) == 0:
            return jsonify({'error': 'Features should be a non-empty list of numbers.'}), 400

        features = np.array(features)

        if features.ndim == 1:
            features = features.reshape(1, -1)
        elif features.ndim != 2 or features.shape[1] != 60:
            return jsonify({'error': 'Features array should be of shape (1, 60).'}), 400

        # Load the model
        model = pickle.load(open('model.pkl', 'rb'))

        # Make prediction
        prediction = model.predict(features)

        return jsonify({'prediction': 'Mine' if prediction[0] == 1 else 'Rock'})

    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
