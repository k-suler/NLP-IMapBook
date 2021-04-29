from sklearn.model_selection import train_test_split

col_to_predict = 'CodePreliminary'


def get_classes(df):
    """ Return all distinct classes """
    classes = df.CodePreliminary.unique()
    return classes


def split_train_test(df):
    """ Split the data to train and test set """
    X = df[['Message']]
    y = df[[col_to_predict]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

