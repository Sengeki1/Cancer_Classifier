from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Classifier:
    def __init__(self, input, ouput):
        self.input_train, self.input_test, self.output_train, self.output_test = train_test_split(
            input,
            ouput,
            test_size=0.3,
            random_state=99
        )
        self.clf = DecisionTreeClassifier(random_state = 1)

    def train(self):
        self.clf.fit(self.input_train, self.output_train)
    
    def predict(self):
        output_predict = self.clf.predict(self.input_test)
        return accuracy_score(self.output_test, output_predict)
