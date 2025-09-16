"""
Configurações do projeto Meta Ads Dashboard
"""
import os
from dotenv import load_dotenv
import logging

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Configurações principais do aplicativo"""

    # Meta Business API
    META_APP_ID = os.getenv('META_APP_ID')
    META_APP_SECRET = os.getenv('META_APP_SECRET')
    META_ACCESS_TOKEN = os.getenv('META_ACCESS_TOKEN')
    META_AD_ACCOUNT_ID = os.getenv('META_AD_ACCOUNT_ID')

    # Configurações do app
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    @classmethod
    def validate_credentials(cls):
        """Valida se todas as credenciais necessárias estão configuradas"""
        credentials = {
            'META_APP_ID': cls.META_APP_ID,
            'META_APP_SECRET': cls.META_APP_SECRET,
            'META_ACCESS_TOKEN': cls.META_ACCESS_TOKEN,
            'META_AD_ACCOUNT_ID': cls.META_AD_ACCOUNT_ID
        }

        missing = []
        for key, value in credentials.items():
            if not value:
                missing.append(key)
                logger.error(f"Variável de ambiente {key} não encontrada")

        if missing:
            logger.error(f"Credenciais faltando: {', '.join(missing)}")
            return False

        # Valida formato do AD_ACCOUNT_ID
        if not cls.META_AD_ACCOUNT_ID.startswith('act_'):
            logger.warning(
                f"AD_ACCOUNT_ID deve começar com 'act_': {cls.META_AD_ACCOUNT_ID}")
            # Adiciona o prefixo se não existir
            cls.META_AD_ACCOUNT_ID = f"act_{cls.META_AD_ACCOUNT_ID}"

        logger.info("Todas as credenciais foram carregadas com sucesso")
        return True

    @classmethod
    def print_config(cls):
        """Imprime configuração atual (sem mostrar valores sensíveis)"""
        print("=== CONFIGURAÇÃO ATUAL ===")
        print(
            f"META_APP_ID: {'✅ Configurado' if cls.META_APP_ID else '❌ Não encontrado'}")
        print(
            f"META_APP_SECRET: {'✅ Configurado' if cls.META_APP_SECRET else '❌ Não encontrado'}")
        print(
            f"META_ACCESS_TOKEN: {'✅ Configurado' if cls.META_ACCESS_TOKEN else '❌ Não encontrado'}")
        print(
            f"META_AD_ACCOUNT_ID: {'✅ Configurado' if cls.META_AD_ACCOUNT_ID else '❌ Não encontrado'}")
        print(f"DEBUG: {cls.DEBUG}")
        print("========================")

    @classmethod
    def get_debug_info(cls):
        """Retorna informações de debug sem expor dados sensíveis"""
        return {
            'app_id_configured': bool(cls.META_APP_ID),
            'app_secret_configured': bool(cls.META_APP_SECRET),
            'access_token_configured': bool(cls.META_ACCESS_TOKEN),
            'ad_account_id_configured': bool(cls.META_AD_ACCOUNT_ID),
            'app_id_length': len(cls.META_APP_ID) if cls.META_APP_ID else 0,
            'access_token_length': len(cls.META_ACCESS_TOKEN) if cls.META_ACCESS_TOKEN else 0,
            'ad_account_id_format': cls.META_AD_ACCOUNT_ID[:4] + '...' if cls.META_AD_ACCOUNT_ID else None
        }


# Configurações específicas para modelos
MODEL_CONFIG = {
    'arima': {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 7)
    },
    'tensorflow': {
        'epochs': 100,
        'batch_size': 32,
        'validation_split': 0.2
    },
    'linear_regression': {
        'test_size': 0.2,
        'random_state': 42
    }
}
