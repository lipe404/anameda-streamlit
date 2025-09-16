# Conexão com Meta Business API
"""
Módulo para conexão com a API do Meta Business Suite
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.campaign import Campaign
from facebook_business.adobjects.adset import AdSet
from facebook_business.adobjects.ad import Ad
from config.settings import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaAdsAPI:
    """Classe para interação com a API do Meta Business Suite"""

    def __init__(self):
        """Inicializa a conexão com a API"""
        self.app_id = Config.META_APP_ID
        self.app_secret = Config.META_APP_SECRET
        self.access_token = Config.META_ACCESS_TOKEN
        self.ad_account_id = Config.META_AD_ACCOUNT_ID

        # Inicializa a API
        FacebookAdsApi.init(
            app_id=self.app_id,
            app_secret=self.app_secret,
            access_token=self.access_token
        )

        self.ad_account = AdAccount(self.ad_account_id)

    def get_campaigns(self, status='ACTIVE'):
        """
        Recupera campanhas ativas da conta de anúncios

        Args:
            status (str): Status das campanhas ('ACTIVE', 'PAUSED', etc.)

        Returns:
            list: Lista de campanhas
        """
        try:
            campaigns = self.ad_account.get_campaigns(
                fields=[
                    'id',
                    'name',
                    'status',
                    'objective',
                    'created_time',
                    'updated_time',
                    'start_time',
                    'stop_time'
                ],
                params={'effective_status': [status]}
            )
            return list(campaigns)
        except Exception as e:
            logger.error(f"Erro ao buscar campanhas: {e}")
            return []

    def get_campaign_insights(self, campaign_id, date_preset='last_30d'):
        """
        Recupera insights de uma campanha específica

        Args:
            campaign_id (str): ID da campanha
            date_preset (str): Período dos dados

        Returns:
            dict: Dados de insights da campanha
        """
        try:
            campaign = Campaign(campaign_id)
            insights = campaign.get_insights(
                fields=[
                    'campaign_id',
                    'campaign_name',
                    'impressions',
                    'reach',
                    'clicks',
                    'spend',
                    'cpm',
                    'cpc',
                    'ctr',
                    'frequency',
                    'actions',
                    'cost_per_action_type',
                    'video_views',
                    'video_view_rate',
                    'date_start',
                    'date_stop'
                ],
                params={
                    'date_preset': date_preset,
                    'time_increment': 1
                }
            )
            return list(insights)
        except Exception as e:
            logger.error(
                f"Erro ao buscar insights da campanha {campaign_id}: {e}")
            return []

    def get_all_campaigns_data(self):
        """
        Recupera dados completos de todas as campanhas ativas

        Returns:
            pd.DataFrame: DataFrame com dados das campanhas
        """
        campaigns = self.get_campaigns()
        all_data = []

        for campaign in campaigns:
            insights = self.get_campaign_insights(campaign['id'])

            for insight in insights:
                data = {
                    'campaign_id': insight.get('campaign_id'),
                    'campaign_name': insight.get('campaign_name'),
                    'date': insight.get('date_start'),
                    'impressions': int(insight.get('impressions', 0)),
                    'reach': int(insight.get('reach', 0)),
                    'clicks': int(insight.get('clicks', 0)),
                    'spend': float(insight.get('spend', 0)),
                    'cpm': float(insight.get('cpm', 0)),
                    'cpc': float(insight.get('cpc', 0)),
                    'ctr': float(insight.get('ctr', 0)),
                    'frequency': float(insight.get('frequency', 0)),
                }

                # Processa ações (conversões)
                actions = insight.get('actions', [])
                for action in actions:
                    action_type = action.get('action_type')
                    if action_type == 'page_engagement':
                        data['page_engagement'] = int(action.get('value', 0))
                    elif action_type == 'post_engagement':
                        data['post_engagement'] = int(action.get('value', 0))
                    elif action_type == 'link_click':
                        data['link_clicks'] = int(action.get('value', 0))

                all_data.append(data)

        df = pd.DataFrame(all_data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

        return df

    def get_account_insights(self):
        """
        Recupera insights gerais da conta

        Returns:
            dict: Insights da conta
        """
        try:
            insights = self.ad_account.get_insights(
                fields=[
                    'impressions',
                    'reach',
                    'clicks',
                    'spend',
                    'cpm',
                    'cpc',
                    'ctr',
                    'frequency'
                ],
                params={'date_preset': 'last_30d'}
            )
            return list(insights)[0] if insights else {}
        except Exception as e:
            logger.error(f"Erro ao buscar insights da conta: {e}")
            return {}
