# Modelo ARIMA
"""
Modelo ARIMA para previsão de séries temporais
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


class ARIMAPredictor:
    """Classe para previsões usando modelo ARIMA"""

    def __init__(self, order=(1, 1, 1)):
        """
        Inicializa o modelo ARIMA

        Args:
            order (tuple): Ordem do modelo ARIMA (p, d, q)
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.data = None

    def prepare_data(self, data: pd.DataFrame, target_column: str, date_column: str = 'date'):
        """
        Prepara os dados para o modelo ARIMA

        Args:
            data (pd.DataFrame): Dados de entrada
            target_column (str): Coluna alvo para previsão
            date_column (str): Coluna de data
        """
        # Agrupa por data se necessário
        if data[date_column].duplicated().any():
            data = data.groupby(date_column)[target_column].sum().reset_index()

        # Ordena por data
        data = data.sort_values(date_column)

        # Define índice como data
        data.set_index(date_column, inplace=True)

        # Remove valores nulos
        data = data.dropna()

        self.data = data[target_column]

    def check_stationarity(self):
        """
        Verifica se a série é estacionária

        Returns:
            dict: Resultados do teste de estacionariedade
        """
        result = adfuller(self.data)

        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }

    def fit(self):
        """
        Treina o modelo ARIMA

        Returns:
            dict: Métricas do modelo ajustado
        """
        try:
            self.model = ARIMA(self.data, order=self.order)
            self.fitted_model = self.model.fit()

            return {
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic,
                'success': True
            }
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }

    def predict(self, steps: int = 7):
        """
        Faz previsões para os próximos períodos

        Args:
            steps (int): Número de períodos para prever

        Returns:
            dict: Previsões e intervalos de confiança
        """
        if self.fitted_model is None:
            raise ValueError(
                "Modelo não foi treinado. Execute fit() primeiro.")

        # Faz previsões
        forecast = self.fitted_model.forecast(steps=steps)
        conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()

        # Cria datas futuras
        last_date = self.data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )

        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted': forecast.values,
            'lower_ci': conf_int.iloc[:, 0].values,
            'upper_ci': conf_int.iloc[:, 1].values
        })

        return predictions_df

    def get_model_summary(self):
        """
        Retorna resumo do modelo

        Returns:
            str: Resumo do modelo
        """
        if self.fitted_model is None:
            return "Modelo não foi treinado."

        return str(self.fitted_model.summary())

    def decompose_series(self):
        """
        Decompõe a série temporal em tendência, sazonalidade e resíduo

        Returns:
            dict: Componentes da decomposição
        """
        if len(self.data) < 14:  # Mínimo para decomposição
            return None

        decomposition = seasonal_decompose(
            self.data,
            model='additive',
            period=7  # Assumindo dados diários com padrão semanal
        )

        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'observed': decomposition.observed
        }
