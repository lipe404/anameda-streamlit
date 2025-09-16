"""
Módulo para análise diagnóstica dos dados de campanhas
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DiagnosticAnalytics:
    """Classe para análises diagnósticas"""

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa com os dados das campanhas

        Args:
            data (pd.DataFrame): Dados das campanhas
        """
        self.data = data

    def analyze_performance_trends(self):
        """
        Analisa tendências de desempenho ao longo do tempo

        Returns:
            dict: Análise de tendências
        """
        trends = {}

        # Agrupa dados por data
        daily_metrics = self.data.groupby('date').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'ctr': 'mean',
            'cpc': 'mean',
            'cpm': 'mean'
        }).reset_index()

        # Calcula correlações com o tempo
        daily_metrics['days_since_start'] = (
            daily_metrics['date'] - daily_metrics['date'].min()).dt.days

        for metric in ['spend', 'impressions', 'clicks', 'ctr', 'cpc', 'cpm']:
            correlation, p_value = stats.pearsonr(
                daily_metrics['days_since_start'], daily_metrics[metric])

            trends[metric] = {
                'correlation_with_time': correlation,
                'p_value': p_value,
                'trend': 'crescente' if correlation > 0.1 else 'decrescente' if correlation < -0.1 else 'estável',
                'significance': 'significativo' if p_value < 0.05 else 'não significativo'
            }

        return trends

    def identify_anomalies(self, metric='spend', threshold=2):
        """
        Identifica anomalias nos dados usando Z-score

        Args:
            metric (str): Métrica para análise de anomalias
            threshold (float): Limiar para detecção de anomalias

        Returns:
            pd.DataFrame: Dados com anomalias identificadas
        """
        data_with_anomalies = self.data.copy()

        # Calcula Z-score
        z_scores = np.abs(stats.zscore(data_with_anomalies[metric]))
        data_with_anomalies['is_anomaly'] = z_scores > threshold
        data_with_anomalies['z_score'] = z_scores

        return data_with_anomalies

    def correlation_analysis(self):
        """
        Analisa correlações entre métricas

        Returns:
            pd.DataFrame: Matriz de correlação
        """
        numeric_columns = ['spend', 'impressions',
                           'clicks', 'ctr', 'cpc', 'cpm', 'frequency']
        available_columns = [
            col for col in numeric_columns if col in self.data.columns]

        correlation_matrix = self.data[available_columns].corr()
        return correlation_matrix

    def campaign_efficiency_analysis(self):
        """
        Analisa eficiência das campanhas

        Returns:
            pd.DataFrame: Análise de eficiência
        """
        campaign_analysis = self.data.groupby('campaign_name').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'ctr': 'mean',
            'cpc': 'mean',
            'cpm': 'mean'
        }).reset_index()

        # Calcula métricas de eficiência
        campaign_analysis['cost_per_impression'] = campaign_analysis['spend'] / \
            campaign_analysis['impressions']
        campaign_analysis['efficiency_score'] = (
            campaign_analysis['ctr'] / campaign_analysis['cpc']
        ).fillna(0)

        # Classifica campanhas por eficiência
        campaign_analysis['efficiency_quartile'] = pd.qcut(
            campaign_analysis['efficiency_score'],
            q=4,
            labels=['Baixa', 'Média-Baixa', 'Média-Alta', 'Alta']
        )

        return campaign_analysis

    def seasonal_analysis(self):
        """
        Analisa padrões sazonais nos dados

        Returns:
            dict: Análise sazonal
        """
        seasonal_data = self.data.copy()
        seasonal_data['day_of_week'] = seasonal_data['date'].dt.day_name()
        seasonal_data['hour'] = seasonal_data['date'].dt.hour
        seasonal_data['month'] = seasonal_data['date'].dt.month

        # Análise por dia da semana
        weekly_performance = seasonal_data.groupby('day_of_week').agg({
            'spend': 'mean',
            'impressions': 'mean',
            'clicks': 'mean',
            'ctr': 'mean'
        })

        # Análise por mês
        monthly_performance = seasonal_data.groupby('month').agg({
            'spend': 'mean',
            'impressions': 'mean',
            'clicks': 'mean',
            'ctr': 'mean'
        })

        return {
            'weekly': weekly_performance,
            'monthly': monthly_performance
        }

    def create_diagnostic_charts(self):
        """
        Cria gráficos para análise diagnóstica

        Returns:
            dict: Gráficos diagnósticos
        """
        charts = {}

        # Gráfico de correlação
        correlation_matrix = self.correlation_analysis()
        charts['correlation_heatmap'] = px.imshow(
            correlation_matrix,
            title='Matriz de Correlação entre Métricas',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )

        # Gráfico de anomalias
        anomaly_data = self.identify_anomalies()
        charts['anomaly_detection'] = px.scatter(
            anomaly_data,
            x='date',
            y='spend',
            color='is_anomaly',
            title='Detecção de Anomalias nos Gastos',
            labels={'spend': 'Gasto (R\$)', 'date': 'Data'}
        )

        # Análise de eficiência
        efficiency_data = self.campaign_efficiency_analysis()
        charts['efficiency_analysis'] = px.scatter(
            efficiency_data,
            x='cpc',
            y='ctr',
            size='spend',
            color='efficiency_quartile',
            title='Análise de Eficiência das Campanhas',
            labels={'cpc': 'CPC (R\$)', 'ctr': 'CTR (%)'}
        )

        return charts
