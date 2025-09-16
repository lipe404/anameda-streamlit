"""
Módulo para análise descritiva dos dados de campanhas
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DescriptiveAnalytics:
    """Classe para análises descritivas"""

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa com os dados das campanhas

        Args:
            data (pd.DataFrame): Dados das campanhas
        """
        self.data = data

    def get_summary_statistics(self):
        """
        Calcula estatísticas descritivas básicas

        Returns:
            dict: Estatísticas resumidas
        """
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        summary = {}

        for col in numeric_columns:
            summary[col] = {
                'mean': self.data[col].mean(),
                'median': self.data[col].median(),
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'total': self.data[col].sum()
            }

        return summary

    def get_campaign_performance_overview(self):
        """
        Gera visão geral do desempenho das campanhas

        Returns:
            dict: Métricas de desempenho
        """
        if self.data.empty:
            return {}

        total_spend = self.data['spend'].sum()
        total_impressions = self.data['impressions'].sum()
        total_clicks = self.data['clicks'].sum()
        total_reach = self.data['reach'].sum()

        avg_ctr = self.data['ctr'].mean()
        avg_cpc = self.data['cpc'].mean()
        avg_cpm = self.data['cpm'].mean()

        return {
            'total_spend': total_spend,
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'total_reach': total_reach,
            'avg_ctr': avg_ctr,
            'avg_cpc': avg_cpc,
            'avg_cpm': avg_cpm,
            'campaigns_count': self.data['campaign_name'].nunique(),
            'date_range': f"{self.data['date'].min().strftime('%Y-%m-%d')} a {self.data['date'].max().strftime('%Y-%m-%d')}"
        }

    def create_performance_charts(self):
        """
        Cria gráficos de desempenho

        Returns:
            dict: Dicionário com gráficos Plotly
        """
        charts = {}

        # Gráfico de gastos ao longo do tempo
        daily_spend = self.data.groupby('date')['spend'].sum().reset_index()
        charts['spend_timeline'] = px.line(
            daily_spend,
            x='date',
            y='spend',
            title='Gastos Diários das Campanhas',
            labels={'spend': 'Gasto (R\$)', 'date': 'Data'}
        )

        # Gráfico de impressões vs clicks
        charts['impressions_clicks'] = px.scatter(
            self.data,
            x='impressions',
            y='clicks',
            color='campaign_name',
            size='spend',
            title='Impressões vs Clicks por Campanha',
            labels={'impressions': 'Impressões', 'clicks': 'Clicks'}
        )

        # Gráfico de CTR por campanha
        campaign_ctr = self.data.groupby('campaign_name')[
            'ctr'].mean().reset_index()
        charts['ctr_by_campaign'] = px.bar(
            campaign_ctr,
            x='campaign_name',
            y='ctr',
            title='CTR Médio por Campanha',
            labels={'ctr': 'CTR (%)', 'campaign_name': 'Campanha'}
        )

        # Gráfico de distribuição de gastos
        charts['spend_distribution'] = px.histogram(
            self.data,
            x='spend',
            title='Distribuição de Gastos Diários',
            labels={'spend': 'Gasto (R\$)', 'count': 'Frequência'}
        )

        return charts

    def get_top_performing_campaigns(self, metric='ctr', top_n=5):
        """
        Identifica as campanhas com melhor desempenho

        Args:
            metric (str): Métrica para ranking
            top_n (int): Número de campanhas no top

        Returns:
            pd.DataFrame: Top campanhas
        """
        campaign_performance = self.data.groupby('campaign_name').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'ctr': 'mean',
            'cpc': 'mean',
            'cpm': 'mean'
        }).reset_index()

        return campaign_performance.nlargest(top_n, metric)
