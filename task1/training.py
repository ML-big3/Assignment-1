import config
import functions
import pandas as pd
from sklearn.utils import shuffle
import zipfile

# training the Sum Without Noise dataset
def train_sum_without_noise_data():
    print("Starting SUM Without Noisy")
    dataset = pd.read_csv(config.SUM_WO_NOISE_DS, sep = ';')
    X = dataset[['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 6',
                        'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']].values
    y = dataset['Target'].values
    functions.linear_regression(X, y)
    functions.decision_tree_regression(X, y)
    y = dataset['Target Class'].values
    y = pd.Series(y)
    functions.logistic_regression_with_split_metrics(X, y)
    functions.knn_with_split_metrics(X, y)
    

# training the Sum With Noise dataset
def train_sum_with_noise_data():
    print("Starting SUM With Noisy")
    dataset = pd.read_csv(config.SUM_WI_NOISE_DS, sep = ';')
    X = dataset[['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 5 (meaningless but please still use it)','Feature 6',
                        'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']].values
    y = dataset['Noisy Target'].values
    functions.linear_regression(X, y)
    functions.decision_tree_regression(X, y)
    y = dataset['Noisy Target Class'].values
    y = pd.Series(y)
    functions.logistic_regression_with_split_metrics(X, y)
    functions.knn_with_split_metrics(X, y)
    


# training the Fashion Mnist dataset
def train_fashion_mnist_data():
    print("Fashion Mnist dataset")
    with zipfile.ZipFile(config.FASHION_MNIST_TRAIN_ZIP,"r") as zip_ref:
        zip_ref.extractall(config.FASHION_MNIST_TRAIN_FOLDER)
    data_train = pd.read_csv(config.FASHION_MNIST_TRAIN_TRAIN, sep=",")

    features = []
    for i in range(1, 785):
        features.append("pixel" + str(i))

    X = data_train[features].values
    y = data_train["label"].values
    functions.linear_regression(X, y)
    functions.decision_tree_regression(X, y)
    functions.logistic_regression_with_split_metrics(X, y)
    functions.knn_with_split_metrics(X, y)


# training the Skin_Non_Skin dataset
def train_skin_data():
    print("Start Skin dataset")
    data = pd.read_csv(config.SKIN_NO_SKIN, sep="\t", names=["a", "b", "c", "d"])
    X = data[["a", "b", "c"]].values
    y = data["d"].values
    X, y = shuffle(X, y)
    functions.linear_regression(X, y)
    functions.decision_tree_regression(X, y)
    functions.logistic_regression(X, y)
    functions.knn(X, y)

train_sum_without_noise_data()
train_sum_with_noise_data()
train_skin_data()
train_fashion_mnist_data()
