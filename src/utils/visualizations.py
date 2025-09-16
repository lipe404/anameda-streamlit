"""
Utilitários para criação de visualizações
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class VisualizationUtils:
    """Classe para utilitários de visualização"""

    @staticmethod
    def create_kpi_cards(data: dict) -> dict:
        """
        Cria dados para cards de KPI

        Args:
            data (dict): Dados de performance

        Returns:
            dict: Dados formatados para cards
        """
        return {
            'total_spend': {
                'value': f"R\$ {data.get('total_spend', 0):,.2f}",
                'label': 'Gasto Total',
                'delta': '+5.2%',
                'color': 'blue'
            },
            'total_impressions': {
                'value': f"{data.get('total_impressions', 0):,}",
                'label': 'Impressões',
                'delta': '+12.3%',
                'color': 'green'
            },
            'total_clicks': {
                'value': f"{data.get('total_clicks', 0):,}",
                'label': 'Clicks',
                'delta': '+8.7%',
                'color': 'orange'
            },
            'avg_ctr': {
                'value': f"{data.get('avg_ctr', 0):.2f}%",
                'label': 'CTR Médio',
                'delta': '+2.1%',
                'color': 'purple'
            }
        }

    @staticmethod
    def create_performance_dashboard(data: pd.DataFrame) -> dict:
        """
        Cria dashboard completo de performance

        Args:
            data (pd.DataFrame): Dados das campanhas

        Returns:
            dict: Gráficos do dashboard
        """
        charts = {}

        if data.empty:
            return charts

        # Gráfico de gastos ao longo do tempo
        daily_spend = data.groupby('date')['spend'].sum().reset_index()
        charts['spend_timeline'] = px.line(
            daily_spend,
            x='date',
            y='spend',
            title='Evolução dos Gastos Diários',
            labels={'spend': 'Gasto (R\$)', 'date': 'Data'}
        )
        charts['spend_timeline'].update_traces(line_color='#1f77b4')

        # Gráfico de performance por campanha
        campaign_perf = data.groupby('campaign_name').agg({
            'spend': 'sum',
            'clicks': 'sum',
            'impressions': 'sum',
            'ctr': 'mean'
        }).reset_index()

        charts['campaign_performance'] = px.bar(
            campaign_perf,
            x='campaign_name',
            y='spend',
            title='Gasto por Campanha',
            labels={'spend': 'Gasto (R\$)', 'campaign_name': 'Campanha'}
        )
        charts['campaign_performance'].update_traces(marker_color='#ff7f0e')

        # Scatter plot CTR vs CPC
        charts['ctr_vs_cpc'] = px.scatter(
            data,
            x='cpc',
            y='ctr',
            size='spend',
            color='campaign_name',
            title='CTR vs CPC por Campanha',
            labels={'cpc': 'CPC (R\$)', 'ctr': 'CTR (%)'}
        )

        # Distribuição de CTR
        charts['ctr_distribution'] = px.histogram(
            data,
            x='ctr',
            title='Distribuição do CTR',
            labels={'ctr': 'CTR (%)', 'count': 'Frequência'},
            nbins=20
        )
        charts['ctr_distribution'].update_traces(marker_color='#2ca02c')

        return charts

    @staticmethod
    def create_comparison_chart(data: pd.DataFrame, x_col: str, y_col: str,
                                color_col: str = None, title: str = None) -> go.Figure:
        """
        Cria gráfico de comparação genérico

        Args:
            data (pd.DataFrame): Dados
            x_col (str): Coluna X
            y_col (str): Coluna Y
            color_col (str): Coluna para cor
            title (str): Título do gráfico

        Returns:
            go.Figure: Gráfico Plotly
        """
        if color_col:
            fig = px.scatter(
                data, x=x_col, y=y_col, color=color_col,
                title=title or f'{y_col} vs {x_col}'
            )
        else:
            fig = px.scatter(
                data, x=x_col, y=y_col,
                title=title or f'{y_col} vs {x_col}'
            )

        return fig

    @staticmethod
    def create_trend_analysis(data: pd.DataFrame, date_col: str,
                              metric_col: str, title: str = None) -> go.Figure:
        """
        Cria análise de tendência

        Args:
            data (pd.DataFrame): Dados
            date_col (str): Coluna de data
            metric_col (str): Métrica para análise
            title (str): Título

        Returns:
            go.Figure: Gráfico de tendência
        """
        # Agrupa por data
        daily_data = data.groupby(date_col)[metric_col].sum().reset_index()

        # Calcula média móvel
        daily_data['ma_7'] = daily_data[metric_col].rolling(window=7).mean()

        fig = go.Figure()

        # Dados originais
        fig.add_trace(go.Scatter(
            x=daily_data[date_col],
            y=daily_data[metric_col],
            mode='lines+markers',
            name='Dados Diários',
            line=dict(color='lightblue')
        ))

        # Média móvel
        fig.add_trace(go.Scatter(
            x=daily_data[date_col],
            y=daily_data['ma_7'],
            mode='lines',
            name='Média Móvel (7 dias)',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title=title or f'Tendência de {metric_col}',
            xaxis_title='Data',
            yaxis_title=metric_col,
            hovermode='x unified'
        )

        return fig

    @staticmethod
    def create_correlation_heatmap(data: pd.DataFrame, columns: list = None) -> go.Figure:
        """
        Cria heatmap de correlação

        Args:
            data (pd.DataFrame): Dados
            columns (list): Colunas para análise

        Returns:
            go.Figure: Heatmap de correlação
        """
        if columns is None:
            columns = ['spend', 'impressions', 'clicks', 'ctr', 'cpc', 'cpm']

        # Filtra colunas existentes
        available_columns = [col for col in columns if col in data.columns]

        if len(available_columns) < 2:
            return go.Figure()

        # Calcula correlação
        corr_matrix = data[available_columns].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title='Matriz de Correlação entre Métricas',
            width=600,
            height=500
        )

        return fig
