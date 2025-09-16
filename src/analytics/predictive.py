"""
Módulo para análise preditiva integrando todos os modelos
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.models.arima_model import ARIMAPredictor
from src.models.tensorflow_model import TensorFlowPredictor
from src.models.linear_regression import LinearRegressionPredictor


class PredictiveAnalytics:
    """Classe para análises preditivas usando múltiplos modelos"""

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa com os dados das campanhas

        Args:
            data (pd.DataFrame): Dados das campanhas
        """
        self.data = data
        self.models = {}
        self.predictions = {}

    def prepare_time_series_data(self, target_column: str):
        """
        Prepara dados de série temporal

        Args:
            target_column (str): Coluna alvo para previsão

        Returns:
            pd.DataFrame: Dados agregados por data
        """
        # Agrupa por data
        daily_data = self.data.groupby(
            'date')[target_column].sum().reset_index()
        daily_data = daily_data.sort_values('date')

        return daily_data

    def train_arima_model(self, target_column: str, order=(1, 1, 1)):
        """
        Treina modelo ARIMA

        Args:
            target_column (str): Coluna alvo
            order (tuple): Ordem do modelo ARIMA

        Returns:
            dict: Resultados do treinamento
        """
        daily_data = self.prepare_time_series_data(target_column)

        if len(daily_data) < 10:
            return {'error': 'Dados insuficientes para ARIMA', 'success': False}

        arima_model = ARIMAPredictor(order=order)
        arima_model.prepare_data(daily_data, target_column)

        # Verifica estacionariedade
        stationarity = arima_model.check_stationarity()

        # Treina modelo
        fit_results = arima_model.fit()

        if fit_results['success']:
            self.models[f'arima_{target_column}'] = arima_model
            return {
                'model_type': 'ARIMA',
                'target': target_column,
                'stationarity': stationarity,
                'fit_results': fit_results,
                'success': True
            }

        return fit_results

    def train_tensorflow_model(self, target_column: str, sequence_length=7, epochs=50):
        """
        Treina modelo TensorFlow

        Args:
            target_column (str): Coluna alvo
            sequence_length (int): Comprimento da sequência
            epochs (int): Número de épocas

        Returns:
            dict: Resultados do treinamento
        """
        daily_data = self.prepare_time_series_data(target_column)

        if len(daily_data) < sequence_length + 10:
            return {'error': 'Dados insuficientes para TensorFlow', 'success': False}

        tf_model = TensorFlowPredictor(
            sequence_length=sequence_length,
            epochs=epochs,
            batch_size=16
        )
        tf_model.prepare_data(daily_data, target_column)

        # Treina modelo
        fit_results = tf_model.fit()

        if fit_results['success']:
            self.models[f'tensorflow_{target_column}'] = tf_model
            return {
                'model_type': 'TensorFlow',
                'target': target_column,
                'fit_results': fit_results,
                'success': True
            }

        return fit_results

    def train_linear_regression_model(self, target_column: str, model_type='ridge'):
        """
        Treina modelo de Regressão Linear

        Args:
            target_column (str): Coluna alvo
            model_type (str): Tipo do modelo

        Returns:
            dict: Resultados do treinamento
        """
        if len(self.data) < 20:
            return {'error': 'Dados insuficientes para Regressão Linear', 'success': False}

        lr_model = LinearRegressionPredictor(model_type=model_type)

        # Treina modelo
        fit_results = lr_model.fit(self.data, target_column)

        if fit_results['success']:
            self.models[f'linear_{target_column}'] = lr_model
            return {
                'model_type': 'Linear Regression',
                'target': target_column,
                'fit_results': fit_results,
                'success': True
            }

        return fit_results

    def train_all_models(self, target_column: str):
        """
        Treina todos os modelos para uma métrica

        Args:
            target_column (str): Coluna alvo

        Returns:
            dict: Resultados de todos os modelos
        """
        results = {}

        # ARIMA
        try:
            results['arima'] = self.train_arima_model(target_column)
        except Exception as e:
            results['arima'] = {'error': str(e), 'success': False}

        # TensorFlow
        try:
            results['tensorflow'] = self.train_tensorflow_model(target_column)
        except Exception as e:
            results['tensorflow'] = {'error': str(e), 'success': False}

        # Linear Regression
        try:
            results['linear_regression'] = self.train_linear_regression_model(
                target_column)
        except Exception as e:
            results['linear_regression'] = {'error': str(e), 'success': False}

        return results

    def generate_predictions(self, target_column: str, steps: int = 7):
        """
        Gera previsões usando todos os modelos treinados

        Args:
            target_column (str): Coluna alvo
            steps (int): Número de períodos para prever

        Returns:
            dict: Previsões de todos os modelos
        """
        predictions = {}

        # ARIMA
        arima_key = f'arima_{target_column}'
        if arima_key in self.models:
            try:
                predictions['arima'] = self.models[arima_key].predict(steps)
            except Exception as e:
                predictions['arima'] = {'error': str(e)}

        # TensorFlow
        tf_key = f'tensorflow_{target_column}'
        if tf_key in self.models:
            try:
                predictions['tensorflow'] = self.models[tf_key].predict(steps)
            except Exception as e:
                predictions['tensorflow'] = {'error': str(e)}

        # Linear Regression
        lr_key = f'linear_{target_column}'
        if lr_key in self.models:
            try:
                daily_data = self.prepare_time_series_data(target_column)
                predictions['linear_regression'] = self.models[lr_key].predict(
                    daily_data, target_column, steps
                )
            except Exception as e:
                predictions['linear_regression'] = {'error': str(e)}

        self.predictions[target_column] = predictions
        return predictions

    def create_ensemble_prediction(self, target_column: str, steps: int = 7):
        """
        Cria previsão ensemble combinando todos os modelos

        Args:
            target_column (str): Coluna alvo
            steps (int): Número de períodos

        Returns:
            pd.DataFrame: Previsão ensemble
        """
        predictions = self.generate_predictions(target_column, steps)

        # Coleta previsões válidas
        valid_predictions = []
        model_names = []

        for model_name, pred in predictions.items():
            if isinstance(pred, pd.DataFrame) and 'predicted' in pred.columns:
                valid_predictions.append(pred['predicted'].values)
                model_names.append(model_name)

        if not valid_predictions:
            return pd.DataFrame()

        # Calcula média das previsões
        ensemble_pred = np.mean(valid_predictions, axis=0)

        # Cria DataFrame resultado
        last_date = self.data['date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )

        ensemble_df = pd.DataFrame({
            'date': future_dates,
            'ensemble_prediction': ensemble_pred,
            'models_used': [', '.join(model_names)] * steps
        })

        # Adiciona previsões individuais
        for i, model_name in enumerate(model_names):
            ensemble_df[f'{model_name}_prediction'] = valid_predictions[i]

        return ensemble_df

    def create_prediction_charts(self, target_column: str, steps: int = 7):
        """
        Cria gráficos de previsões

        Args:
            target_column (str): Coluna alvo
            steps (int): Número de períodos

        Returns:
            dict: Gráficos de previsões
        """
        charts = {}

        # Dados históricos
        daily_data = self.prepare_time_series_data(target_column)

        # Gera previsões
        predictions = self.generate_predictions(target_column, steps)
        ensemble = self.create_ensemble_prediction(target_column, steps)

        # Gráfico comparativo de modelos
        fig = go.Figure()

        # Adiciona dados históricos
        fig.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data[target_column],
            mode='lines+markers',
            name='Dados Históricos',
            line=dict(color='blue')
        ))

        # Adiciona previsões de cada modelo
        colors = ['red', 'green', 'orange', 'purple']
        for i, (model_name, pred) in enumerate(predictions.items()):
            if isinstance(pred, pd.DataFrame) and 'predicted' in pred.columns:
                fig.add_trace(go.Scatter(
                    x=pred['date'],
                    y=pred['predicted'],
                    mode='lines+markers',
                    name=f'Previsão {model_name.upper()}',
                    line=dict(color=colors[i % len(colors)], dash='dash')
                ))

        # Adiciona previsão ensemble
        if not ensemble.empty:
            fig.add_trace(go.Scatter(
                x=ensemble['date'],
                y=ensemble['ensemble_prediction'],
                mode='lines+markers',
                name='Previsão Ensemble',
                line=dict(color='black', width=3)
            ))

        fig.update_layout(
            title=f'Previsões para {target_column.title()}',
            xaxis_title='Data',
            yaxis_title=target_column.title(),
            hovermode='x unified'
        )

        charts['predictions_comparison'] = fig

        # Gráfico de intervalos de confiança (ARIMA)
        if 'arima' in predictions and isinstance(predictions['arima'], pd.DataFrame):
            arima_pred = predictions['arima']
            if 'lower_ci' in arima_pred.columns:
                fig_ci = go.Figure()

                # Dados históricos
                fig_ci.add_trace(go.Scatter(
                    x=daily_data['date'],
                    y=daily_data[target_column],
                    mode='lines',
                    name='Dados Históricos',
                    line=dict(color='blue')
                ))

                # Previsão
                fig_ci.add_trace(go.Scatter(
                    x=arima_pred['date'],
                    y=arima_pred['predicted'],
                    mode='lines',
                    name='Previsão ARIMA',
                    line=dict(color='red')
                ))

                # Intervalo de confiança
                fig_ci.add_trace(go.Scatter(
                    x=arima_pred['date'],
                    y=arima_pred['upper_ci'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))

                fig_ci.add_trace(go.Scatter(
                    x=arima_pred['date'],
                    y=arima_pred['lower_ci'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(width=0),
                    name='Intervalo de Confiança'
                ))

                fig_ci.update_layout(
                    title=f'Previsão ARIMA com Intervalo de Confiança - {target_column.title()}',
                    xaxis_title='Data',
                    yaxis_title=target_column.title()
                )

                charts['arima_confidence_interval'] = fig_ci

        return charts

    def get_model_performance_summary(self, target_column: str):
        """
        Retorna resumo de performance dos modelos

        Args:
            target_column (str): Coluna alvo

        Returns:
            dict: Resumo de performance
        """
        summary = {}

        # ARIMA
        arima_key = f'arima_{target_column}'
        if arima_key in self.models:
            model = self.models[arima_key]
            if hasattr(model, 'fitted_model') and model.fitted_model:
                summary['arima'] = {
                    'aic': model.fitted_model.aic,
                    'bic': model.fitted_model.bic,
                    'model_order': model.order
                }

        # TensorFlow
        tf_key = f'tensorflow_{target_column}'
        if tf_key in self.models:
            model = self.models[tf_key]
            history = model.get_training_history()
            if history:
                summary['tensorflow'] = {
                    'final_loss': history['loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'epochs_trained': len(history['loss'])
                }

        # Linear Regression
        lr_key = f'linear_{target_column}'
        if lr_key in self.models:
            try:
                model = self.models[lr_key]
                importance = model.get_feature_importance()

                if importance is not None and not importance.empty:
                    summary['linear_regression'] = {
                        'top_features': importance.head(5).to_dict('records'),
                        'model_type': model.model_type,
                        'total_features': len(importance)
                    }
                else:
                    summary['linear_regression'] = {
                        'model_type': model.model_type,
                        'note': 'Importância das features não disponível'
                    }
            except Exception as e:
                summary['linear_regression'] = {
                    'error': f'Erro ao obter performance: {str(e)}'
                }

        return summary
