# Regressão Linear
"""
Modelo de Regressão Linear para previsão de métricas
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline


class LinearRegressionPredictor:
    """Classe para previsões usando Regressão Linear"""

    def __init__(self, model_type='linear', alpha=1.0, degree=1):
        """
        Inicializa o modelo de regressão

        Args:
            model_type (str): Tipo do modelo ('linear', 'ridge', 'lasso')
            alpha (float): Parâmetro de regularização
            degree (int): Grau para regressão polinomial
        """
        self.model_type = model_type
        self.alpha = alpha
        self.degree = degree
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def prepare_features(self, data: pd.DataFrame, target_column: str, date_column: str = 'date'):
        """
        Prepara features para o modelo

        Args:
            data (pd.DataFrame): Dados de entrada
            target_column (str): Coluna alvo
            date_column (str): Coluna de data
        """
        # Cria cópia dos dados
        df = data.copy()

        # Converte data para features numéricas
        df['day_of_year'] = pd.to_datetime(df[date_column]).dt.dayofyear
        df['day_of_week'] = pd.to_datetime(df[date_column]).dt.dayofweek
        df['month'] = pd.to_datetime(df[date_column]).dt.month
        df['week_of_year'] = pd.to_datetime(
            df[date_column]).dt.isocalendar().week

        # Features de lag (valores anteriores)
        for lag in [1, 3, 7]:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)

        # Features de média móvel
        for window in [3, 7]:
            df[f'{target_column}_ma_{window}'] = df[target_column].rolling(
                window=window).mean()

        # Remove linhas com valores nulos
        df = df.dropna()

        # Seleciona features numéricas (exceto target e date)
        feature_columns = [col for col in df.columns
                           if col not in [target_column, date_column]
                           and df[col].dtype in ['int64', 'float64']]

        self.feature_names = feature_columns

        X = df[feature_columns]
        y = df[target_column]

        return X, y

    def build_model(self):
        """
        Constrói o pipeline do modelo

        Returns:
            Pipeline: Pipeline do modelo
        """
        steps = []

        # Adiciona features polinomiais se degree > 1
        if self.degree > 1:
            steps.append(('poly', PolynomialFeatures(
                degree=self.degree, include_bias=False)))

        # Adiciona normalização
        steps.append(('scaler', StandardScaler()))

        # Adiciona modelo
        if self.model_type == 'ridge':
            steps.append(('regressor', Ridge(alpha=self.alpha)))
        elif self.model_type == 'lasso':
            steps.append(('regressor', Lasso(alpha=self.alpha)))
        else:
            steps.append(('regressor', LinearRegression()))

        return Pipeline(steps)

    def fit(self, data: pd.DataFrame, target_column: str, test_size=0.2, random_state=42):
        """
        Treina o modelo

        Args:
            data (pd.DataFrame): Dados de treinamento
            target_column (str): Coluna alvo
            test_size (float): Proporção dos dados para teste
            random_state (int): Seed para reprodutibilidade

        Returns:
            dict: Métricas de avaliação
        """
        # Prepara features
        X, y = self.prepare_features(data, target_column)

        if len(X) == 0:
            return {
                'error': 'Dados insuficientes após preparação das features',
                'success': False
            }

        # Divide dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Constrói e treina modelo
        self.model = self.build_model()
        self.model.fit(X_train, y_train)

        # Faz previsões
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        # Calcula métricas
        metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'success': True
        }

        return metrics

    def predict(self, data: pd.DataFrame, target_column: str, steps: int = 7):
        """
        Faz previsões para os próximos períodos

        Args:
            data (pd.DataFrame): Dados históricos
            target_column (str): Coluna alvo
            steps (int): Número de períodos para prever

        Returns:
            pd.DataFrame: Previsões
        """
        if self.model is None:
            raise ValueError(
                "Modelo não foi treinado. Execute fit() primeiro.")

        # Prepara dados para previsão
        df = data.copy()
        predictions = []

        # Cria datas futuras
        last_date = pd.to_datetime(df['date']).max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )

        for i, future_date in enumerate(future_dates):
            # Cria linha para previsão
            new_row = df.iloc[-1:].copy()
            new_row['date'] = future_date

            # Atualiza features temporais
            new_row['day_of_year'] = future_date.dayofyear
            new_row['day_of_week'] = future_date.dayofweek
            new_row['month'] = future_date.month
            new_row['week_of_year'] = future_date.isocalendar().week

            # Adiciona à série temporal
            df_extended = pd.concat([df, new_row], ignore_index=True)

            # Prepara features
            X, _ = self.prepare_features(df_extended, target_column)

            if len(X) > 0:
                # Faz previsão
                prediction = self.model.predict(X.iloc[-1:].values)[0]
                predictions.append(prediction)

                # Atualiza dados com previsão
                df.loc[len(df)] = new_row.iloc[0]
                df.loc[len(df)-1, target_column] = prediction
            else:
                predictions.append(0)

        # Cria DataFrame com previsões
        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted': predictions
        })

        return predictions_df

    def get_feature_importance(self):
        """
        Retorna importância das features

        Returns:
            pd.DataFrame: Importância das features
        """
        if self.model is None or self.feature_names is None:
            return None

        # Obtém coeficientes do modelo
        regressor = self.model.named_steps['regressor']

        if hasattr(regressor, 'coef_'):
            coefficients = regressor.coef_

            # Se há features polinomiais, usa apenas os coeficientes originais
            if 'poly' in self.model.named_steps:
                coefficients = coefficients[:len(self.feature_names)]

            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(coefficients)
            }).sort_values('importance', ascending=False)

            return importance_df

        return None
