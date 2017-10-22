import config
import training
import pandas as pd


def strat_training(X, y):
    training.linear_regression(X, y)
    training.svr(X, y)
    training.logistic_regression(X, y)
    training.knn(X, y)

# training the Sum Without Noise dataset
def train_sum_without_noise_data():
    dataset = pd.read_csv(config.SUM_WO_NOISE_DS, sep = ';')
    X = dataset[['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 6',
                        'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']].values
    y = dataset['Target'].values
    strat_training(X, y)

# training the Sum With Noise dataset
def train_sum_with_noise_data():
    dataset = pd.read_csv(config.SUM_WI_NOISE_DS, sep = ';')
    X = dataset[['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 5 (meaningless but please still use it)','Feature 6',
                        'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']].values
    y = dataset['Noisy Target'].values
    strat_training(X, y)

# training the Fashion Mnist dataset
def train_fashion_mnist_data():
    data_train = pd.read_csv(config.FASHION_MNIST_TRAIN, sep=",")
    features = []
    for i in range(1, 785):
        features.append("pixel" + str(i))

    X = data_train[features].values
    y = data_train["label"].values
    strat_training(X, y)

# training the Skin_Non_Skin dataset
def train_skin_data():
    data = pd.read_csv(config.SKIN_NO_SKIN, sep="\t", names=["a", "b", "c", "d"])
    X = data[["a", "b", "c"]].values
    y = data["d"].values
    X, y = shuffle(X, y)
    strat_training(X, y)

train_sum_without_noise_data
train_sum_with_noise_data
train_fashion_mnist_data
train_skin_data