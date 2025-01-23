from src.data_processing_2 import DataPreprocessor
import pandas as pd

# from src.model_training import ModelTrainer
# from src.evaluation import Evaluator
from sklearn.model_selection import train_test_split


def main():
    data = pd.read_csv("data/data.csv")

    X = data.drop("price", axis=1)
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Etapa de pré-processamento
    preprocessor = DataPreprocessor(X_train)
    X_train_processed = preprocessor.preprocess_data()
    #  X_train_processed = src.data_processing_1.preprocess_train_data(X_train)

    print(X_train_processed.head())

    # Etapa de treinamento do modelo
    # model = model_trainer.train(
    #     X_train, y_train, model_name="LinearRegression"
    # )

    # Etapa de avaliação
    # evaluator = Evaluator(model, X_test, y_test)
    # mse, rmse, r2 = evaluator.evaluate()

    # print(f"Model Evaluation Metrics:")
    # print(f"Mean Squared Error (MSE): {mse}")
    # print(f"Root Mean Squared Error (RMSE): {rmse}")
    # print(f"R-squared: {r2}")


if __name__ == "__main__":
    main()
