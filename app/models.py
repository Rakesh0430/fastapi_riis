from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from pydantic import BaseModel

class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisModel:
    def __init__(self):
        self.data = load_iris()
        self.model = self._train_model()

    def _train_model(self):
        X = self.data.data
        y = self.data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

    def predict(self, iris: IrisSpecies):
        data = [
            iris.sepal_length,
            iris.sepal_width,
            iris.petal_length,
            iris.petal_width
        ]
        prediction = self.model.predict([data])
        return self.data.target_names[prediction][0]
