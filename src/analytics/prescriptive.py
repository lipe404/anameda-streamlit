"""
Módulo para análise prescritiva - recomendações baseadas em dados
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class PrescriptiveAnalytics:
    """Classe para análises prescritivas e recomendações"""

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa com os dados das campanhas

        Args:
            data (pd.DataFrame): Dados das campanhas
        """
        self.data = data

    def budget_optimization_recommendations(self):
        """
        Gera recomendações para otimização de orçamento

        Returns:
            dict: Recomendações de orçamento
        """
        # Analisa performance por campanha
        campaign_performance = self.data.groupby('campaign_name').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'ctr': 'mean',
            'cpc': 'mean',
            'cpm': 'mean'
        }).reset_index()

        # Calcula ROI e eficiência
        campaign_performance['cost_per_impression'] = (
            campaign_performance['spend'] / campaign_performance['impressions']
        )
        campaign_performance['efficiency_score'] = (
            campaign_performance['ctr'] / campaign_performance['cpc']
        ).fillna(0)

        # Classifica campanhas
        high_performers = campaign_performance[
            campaign_performance['efficiency_score'] > campaign_performance['efficiency_score'].quantile(
                0.75)
        ]

        low_performers = campaign_performance[
            campaign_performance['efficiency_score'] < campaign_performance['efficiency_score'].quantile(
                0.25)
        ]

        recommendations = {
            'increase_budget': {
                'campaigns': high_performers['campaign_name'].tolist(),
                'reason': 'Alto desempenho e eficiência',
                'suggested_increase': '20-30%'
            },
            'decrease_budget': {
                'campaigns': low_performers['campaign_name'].tolist(),
                'reason': 'Baixo desempenho e eficiência',
                'suggested_decrease': '15-25%'
            },
            'total_budget_reallocation': {
                'from_campaigns': low_performers['campaign_name'].tolist(),
                'to_campaigns': high_performers['campaign_name'].tolist(),
                'potential_savings': low_performers['spend'].sum() * 0.2
            }
        }

        return recommendations

    def bidding_strategy_recommendations(self):
        """
        Gera recomendações para estratégia de lances

        Returns:
            dict: Recomendações de lances
        """
        # Analisa CPC vs CTR
        campaign_analysis = self.data.groupby('campaign_name').agg({
            'cpc': 'mean',
            'ctr': 'mean',
            'spend': 'sum',
            'clicks': 'sum'
        }).reset_index()

        # Identifica oportunidades
        high_cpc_low_ctr = campaign_analysis[
            (campaign_analysis['cpc'] > campaign_analysis['cpc'].quantile(0.75)) &
            (campaign_analysis['ctr'] < campaign_analysis['ctr'].quantile(0.5))
        ]

        low_cpc_high_ctr = campaign_analysis[
            (campaign_analysis['cpc'] < campaign_analysis['cpc'].quantile(0.5)) &
            (campaign_analysis['ctr'] >
             campaign_analysis['ctr'].quantile(0.75))
        ]

        recommendations = {
            'reduce_bids': {
                'campaigns': high_cpc_low_ctr['campaign_name'].tolist(),
                'reason': 'CPC alto com CTR baixo',
                'action': 'Reduzir lances em 10-20%'
            },
            'increase_bids': {
                'campaigns': low_cpc_high_ctr['campaign_name'].tolist(),
                'reason': 'CPC baixo com CTR alto - oportunidade de escalar',
                'action': 'Aumentar lances em 15-25%'
            }
        }

        return recommendations

    def targeting_optimization_recommendations(self):
        """
        Gera recomendações para otimização de segmentação

        Returns:
            dict: Recomendações de targeting
        """
        # Analisa padrões temporais
        self.data['day_of_week'] = pd.to_datetime(
            self.data['date']).dt.day_name()
        self.data['hour'] = pd.to_datetime(self.data['date']).dt.hour

        # Performance por dia da semana
        weekly_performance = self.data.groupby('day_of_week').agg({
            'ctr': 'mean',
            'cpc': 'mean',
            'spend': 'sum'
        })

        best_days = weekly_performance.nlargest(3, 'ctr').index.tolist()
        worst_days = weekly_performance.nsmallest(2, 'ctr').index.tolist()

        recommendations = {
            'schedule_optimization': {
                'increase_budget_days': best_days,
                'decrease_budget_days': worst_days,
                'reason': 'Baseado na performance de CTR por dia da semana'
            },
            'audience_expansion': {
                'high_performing_campaigns': self.data[
                    self.data['ctr'] > self.data['ctr'].quantile(0.8)
                ]['campaign_name'].unique().tolist(),
                'action': 'Expandir audiências similares'
            }
        }

        return recommendations

    def creative_optimization_recommendations(self):
        """
        Gera recomendações para otimização de criativos

        Returns:
            dict: Recomendações de criativos
        """
        # Analisa frequency vs performance
        frequency_analysis = self.data.groupby('campaign_name').agg({
            'frequency': 'mean',
            'ctr': 'mean',
            'cpm': 'mean'
        }).reset_index()

        # Identifica fadiga de criativo
        high_frequency = frequency_analysis[
            frequency_analysis['frequency'] > frequency_analysis['frequency'].quantile(
                0.75)
        ]

        low_ctr_high_freq = high_frequency[
            high_frequency['ctr'] < frequency_analysis['ctr'].median()
        ]

        recommendations = {
            'creative_refresh': {
                'campaigns': low_ctr_high_freq['campaign_name'].tolist(),
                'reason': 'Alta frequência com CTR baixo indica fadiga de criativo',
                'action': 'Criar novos criativos e pausar os atuais'
            },
            'creative_testing': {
                'campaigns': frequency_analysis[
                    frequency_analysis['ctr'] > frequency_analysis['ctr'].quantile(
                        0.75)
                ]['campaign_name'].tolist(),
                'action': 'Testar variações dos criativos de melhor performance'
            }
        }

        return recommendations

    def campaign_clustering_insights(self):
        """
        Agrupa campanhas similares para insights

        Returns:
            dict: Insights de clustering
        """
        # Prepara dados para clustering
        campaign_metrics = self.data.groupby('campaign_name').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'ctr': 'mean',
            'cpc': 'mean',
            'cpm': 'mean',
            'frequency': 'mean'
        }).reset_index()

        # Seleciona features numéricas
        features = ['spend', 'impressions', 'clicks',
                    'ctr', 'cpc', 'cpm', 'frequency']
        X = campaign_metrics[features].fillna(0)

        # Normaliza dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Aplica K-means
        kmeans = KMeans(n_clusters=3, random_state=42)
        campaign_metrics['cluster'] = kmeans.fit_predict(X_scaled)

        # Analisa clusters
        cluster_analysis = {}
        for cluster in range(3):
            cluster_data = campaign_metrics[campaign_metrics['cluster'] == cluster]
            cluster_analysis[f'cluster_{cluster}'] = {
                'campaigns': cluster_data['campaign_name'].tolist(),
                'characteristics': {
                    'avg_spend': cluster_data['spend'].mean(),
                    'avg_ctr': cluster_data['ctr'].mean(),
                    'avg_cpc': cluster_data['cpc'].mean(),
                    'campaign_count': len(cluster_data)
                },
                'recommendations': self._get_cluster_recommendations(cluster_data)
            }

        return cluster_analysis

    def _get_cluster_recommendations(self, cluster_data):
        """
        Gera recomendações específicas para um cluster

        Args:
            cluster_data (pd.DataFrame): Dados do cluster

        Returns:
            list: Lista de recomendações
        """
        recommendations = []

        avg_ctr = cluster_data['ctr'].mean()
        avg_cpc = cluster_data['cpc'].mean()
        avg_spend = cluster_data['spend'].mean()

        if avg_ctr > 2.0:  # CTR alto
            recommendations.append(
                "Cluster de alto desempenho - considere escalar investimento")
        elif avg_ctr < 1.0:  # CTR baixo
            recommendations.append(
                "Cluster de baixo desempenho - revisar targeting e criativos")

        if avg_cpc > 2.0:  # CPC alto
            recommendations.append(
                "CPC elevado - otimizar lances e palavras-chave")

        if avg_spend > 1000:  # Gasto alto
            recommendations.append(
                "Alto investimento - monitorar ROI de perto")

        return recommendations

    def generate_action_plan(self):
        """
        Gera plano de ação consolidado

        Returns:
            dict: Plano de ação completo
        """
        budget_recs = self.budget_optimization_recommendations()
        bidding_recs = self.bidding_strategy_recommendations()
        targeting_recs = self.targeting_optimization_recommendations()
        creative_recs = self.creative_optimization_recommendations()

        action_plan = {
            'immediate_actions': [
                "Pausar campanhas com baixo desempenho identificadas",
                "Aumentar orçamento das campanhas de alto desempenho",
                "Ajustar lances conforme recomendações"
            ],
            'short_term_actions': [
                "Criar novos criativos para campanhas com fadiga",
                "Expandir audiências das campanhas de sucesso",
                "Otimizar cronograma de veiculação"
            ],
            'long_term_actions': [
                "Implementar testes A/B sistemáticos",
                "Desenvolver estratégia de remarketing",
                "Criar dashboard de monitoramento contínuo"
            ],
            'budget_recommendations': budget_recs,
            'bidding_recommendations': bidding_recs,
            'targeting_recommendations': targeting_recs,
            'creative_recommendations': creative_recs
        }

        return action_plan
