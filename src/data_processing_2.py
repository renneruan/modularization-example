from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Classe para pré processamento dos dados, realiza operações de
     preenchimento de valores vazios, normalização (utilizando normalizador
     padrão) e codificação de variáveis categóricas.

    Args:
    - X: DataFrame com features a serem processadas.
    """

    def __init__(self, X):
        self.scaler = StandardScaler()
        self.X = X

    def handle_missing_values(self):
        """
        Preenche valores vazios da coluna "size" com a mediana dos valores
        A mediana foi escolhida devido a presença de outliers.
        """
        size_median = self.X["size"].median()

        X_copy = self.X.copy()
        X_copy["size"] = X_copy["size"].fillna(size_median)
        self.X = X_copy

    def encode_categorical_variables(self):
        """
        Transforma variável categórica de localização em valores numéricos.
        """
        if "location" in self.X.columns:
            X_copy = self.X.copy()
            X_copy["location"] = X_copy["location"].map(
                {"country": 0, "center": 1}
            )
            self.X = X_copy

    def scale_numerical_features(self):
        """
        Normaliza as features numéricas utilizando o normalizador padrão.
        Aplicável apenas aos dados de teste devido ao uso de fit_transform
        """

        # select_dtypes retorna colunas apenas do tipo numérico
        numerical_features = self.X.select_dtypes(
            include=["int64", "float64"]
        ).columns
        X_scaled = self.X.copy()

        X_scaled[numerical_features] = self.scaler.fit_transform(
            X_scaled[numerical_features]
        )

        self.X = X_scaled

    def preprocess_data(self):
        """
        Invoca os métodos para pré-processamento de dados
        Requer que os dados de entrada tenham sido passados no construtor
        """
        self.handle_missing_values()
        self.encode_categorical_variables()
        self.scale_numerical_features()

        return self.X
