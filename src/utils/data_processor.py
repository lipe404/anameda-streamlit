"""
Utilitários para processamento de dados
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataProcessor:
    """Classe para processamento e limpeza de dados"""

    @staticmethod
    def clean_campaign_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e padroniza dados de campanhas

        Args:
            data (pd.DataFrame): Dados brutos

        Returns:
            pd.DataFrame: Dados limpos
        """
        if data.empty:
            return data

        # Cópia dos dados
        cleaned_data = data.copy()

        # Converte colunas de data
        if 'date' in cleaned_data.columns:
            cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])

        # Preenche valores nulos com 0 para métricas numéricas
        numeric_columns = ['impressions', 'clicks',
                           'spend', 'cpm', 'cpc', 'ctr', 'frequency']
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = pd.to_numeric(
                    cleaned_data[col], errors='coerce').fillna(0)

        # Remove linhas com gastos negativos
        if 'spend' in cleaned_data.columns:
            cleaned_data = cleaned_data[cleaned_data['spend'] >= 0]

        # Remove outliers extremos
        cleaned_data = DataProcessor.remove_outliers(cleaned_data)

        return cleaned_data

    @staticmethod
    def remove_outliers(data: pd.DataFrame, columns: list = None, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers dos dados

        Args:
            data (pd.DataFrame): Dados de entrada
            columns (list): Colunas para análise de outliers
            method (str): Método para detecção ('iqr' ou 'zscore')

        Returns:
            pd.DataFrame: Dados sem outliers
        """
        if columns is None:
            columns = ['spend', 'impressions', 'clicks', 'ctr', 'cpc', 'cpm']

        # Filtra apenas colunas existentes
        columns = [col for col in columns if col in data.columns]

        cleaned_data = data.copy()

        for col in columns:
            if method == 'iqr':
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                cleaned_data = cleaned_data[
                    (cleaned_data[col] >= lower_bound) &
                    (cleaned_data[col] <= upper_bound)
                ]

            elif method == 'zscore':
                z_scores = np.abs(
                    (cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std())
                cleaned_data = cleaned_data[z_scores < 3]

        return cleaned_data

    @staticmethod
    def aggregate_daily_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega dados por dia

        Args:
            data (pd.DataFrame): Dados de entrada

        Returns:
            pd.DataFrame: Dados agregados por dia
        """
        if 'date' not in data.columns:
            return data

        # Agrupa por data
        daily_data = data.groupby('date').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'ctr': 'mean',
            'cpc': 'mean',
            'cpm': 'mean',
            'frequency': 'mean',
            'campaign_name': 'nunique'
        }).reset_index()

        # Renomeia coluna de campanhas
        daily_data.rename(
            columns={'campaign_name': 'active_campaigns'}, inplace=True)

        return daily_data

    @staticmethod
    def calculate_derived_metrics(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas derivadas

        Args:
            data (pd.DataFrame): Dados de entrada

        Returns:
            pd.DataFrame: Dados com métricas derivadas
        """
        enhanced_data = data.copy()

        # Cost per mille (CPM) se não existir
        if 'cpm' not in enhanced_data.columns and 'spend' in enhanced_data.columns and 'impressions' in enhanced_data.columns:
            enhanced_data['cpm'] = (
                enhanced_data['spend'] / enhanced_data['impressions'] * 1000).fillna(0)

        # Click-through rate (CTR) se não existir
        if 'ctr' not in enhanced_data.columns and 'clicks' in enhanced_data.columns and 'impressions' in enhanced_data.columns:
            enhanced_data['ctr'] = (
                enhanced_data['clicks'] / enhanced_data['impressions'] * 100).fillna(0)

        # Cost per click (CPC) se não existir
        if 'cpc' not in enhanced_data.columns and 'spend' in enhanced_data.columns and 'clicks' in enhanced_data.columns:
            enhanced_data['cpc'] = (
                enhanced_data['spend'] / enhanced_data['clicks']).fillna(0)

        # Reach rate
        if 'reach' in enhanced_data.columns and 'impressions' in enhanced_data.columns:
            enhanced_data['reach_rate'] = (
                enhanced_data['reach'] / enhanced_data['impressions']).fillna(0)

        # Efficiency score
        if 'ctr' in enhanced_data.columns and 'cpc' in enhanced_data.columns:
            enhanced_data['efficiency_score'] = (
                enhanced_data['ctr'] / enhanced_data['cpc']).fillna(0)

        return enhanced_data

    @staticmethod
    def create_sample_data(days: int = 30) -> pd.DataFrame:
        """
        Cria dados de exemplo para demonstração

        Args:
            days (int): Número de dias de dados

        Returns:
            pd.DataFrame: Dados de exemplo
        """
        np.random.seed(42)

        # Datas
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Campanhas de exemplo
        campaigns = [
            'Campanha Vendas Q4',
            'Campanha Brand Awareness',
            'Campanha Remarketing',
            'Campanha Produto Novo',
            'Campanha Black Friday'
        ]

        sample_data = []

        for date in dates:
            for campaign in campaigns:
                # Simula variação sazonal
                base_multiplier = 1 + 0.3 * \
                    np.sin(2 * np.pi * date.dayofyear / 365)
                weekend_effect = 0.8 if date.weekday() >= 5 else 1.0

                # Métricas base com variação aleatória
                impressions = int(np.random.normal(5000, 1000)
                                  * base_multiplier * weekend_effect)
                impressions = max(impressions, 100)

                ctr = np.random.normal(2.5, 0.5) * base_multiplier
                ctr = max(min(ctr, 10), 0.1)  # Entre 0.1% e 10%

                clicks = int(impressions * ctr / 100)

                cpc = np.random.normal(1.5, 0.3)
                cpc = max(cpc, 0.1)

                spend = clicks * cpc
                cpm = spend / impressions * 1000

                frequency = np.random.normal(1.8, 0.3)
                frequency = max(frequency, 1.0)

                reach = int(impressions / frequency)

                sample_data.append({
                    'date': date,
                    'campaign_id': f'camp_{campaigns.index(campaign)+1}',
                    'campaign_name': campaign,
                    'impressions': impressions,
                    'reach': reach,
                    'clicks': clicks,
                    'spend': round(spend, 2),
                    'cpm': round(cpm, 2),
                    'cpc': round(cpc, 2),
                    'ctr': round(ctr, 2),
                    'frequency': round(frequency, 2)
                })

        return pd.DataFrame(sample_data)
