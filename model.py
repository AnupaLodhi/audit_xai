from sklearn.ensemble import IsolationForest

def train_model(data):
    model = IsolationForest(contamination=0.2, random_state=42)
    model.fit(data)
    scores = model.decision_function(data)
    return model, scores
