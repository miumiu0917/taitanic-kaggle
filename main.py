import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def main():
  data = InputData()
  for i in range(1,100):
    forest = RandomForestClassifier(n_estimators=i)
    forest.fit(data(), data.train_labels)
    train_predict = forest.predict(data())
    print(accuracy(train_predict, data.train_labels))
  
  test_predicts = forest.predict(data(train=False))
  
  with open('./submit.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['PassengerId','Survived'])
    for data, p in zip(data.test_data, test_predicts):
      writer.writerow([data[0], p])


def accuracy(predicts, labels):
  pair = [(predict, label) for predict, label in zip(predicts, labels)]
  percentage = len(list(filter(lambda x: x[0] == x[1], pair))) / len(pair)
  return percentage


class InputData(object):

  def __init__(self, train=True):
    with open('./train.csv', 'r') as f:
      f.readline()
      reader = csv.reader(f)
      data = [x for x in reader]
      self.train_labels = [int(line[1]) for line in data]
      self._train_data = [line[0:1] + line[2:] for line in data]
      self.mean_age = self.mean_age(self._train_data)
      self.GENDER = {'male':0, 'female':1}
      self.EMBARKED = {'S':0, 'C':1, 'Q':2}
    
    with open('./test.csv', 'r') as f:
      f.readline()
      reader = csv.reader(f)
      self.test_data = [x for x in reader]
  

  def __call__(self, train=True):
    if train:
      data = self._train_data
    else:
      data = self.test_data
    return [[self.GENDER[x[3]], int(float(x[4] if x[4] else self.mean_age)), int(x[5]), int(x[6]), self.EMBARKED[x[10]] if x[10] else 0]  for x in data]


  def mean_age(self, lines):
    ages = [int(float(line[4])) for line in lines if line[4]]
    return int(sum(ages) / len(ages) + 0.5)


if __name__ == '__main__':
  main()