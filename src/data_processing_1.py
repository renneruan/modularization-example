from sklearn.preprocessing import StandardScaler


def handle_missing_values(X):
    X_copy = X.copy()
    print(X_copy["size"].median())
    X_copy["size"] = X_copy["size"].fillna(X_copy["size"].median())
    return X_copy


def encode_categorical_variables(X):
    if "location" in X.columns:
        X_copy = X.copy()
        X_copy["location"] = X_copy["location"].map(
            {"country": 0, "center": 1}
        )
    return X_copy


def scale_numerical_features(X):
    scaler = StandardScaler()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
    X_scaled = X.copy()
    X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])
    return X_scaled


def preprocess_train_data(X_train):
    X_train = handle_missing_values(X_train)
    X_train = encode_categorical_variables(X_train)
    X_train = scale_numerical_features(X_train)
    return X_train
