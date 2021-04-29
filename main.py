from preprocess import preprocess_data
from utils import split_train_test

# import pdb

if __name__ == "__main__":
    data = preprocess_data()
    x1, x2, y1, y2 = split_train_test(data)
    # pdb.set_trace()
    print(x1)
