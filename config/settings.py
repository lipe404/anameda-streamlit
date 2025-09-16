# Configurações do projeto
"""
Configurações do projeto Meta Ads Dashboard
"""
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()


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

    # Validação de credenciais
    @classmethod
    def validate_credentials(cls):
        """Valida se todas as credenciais necessárias estão configuradas"""
        required_vars = [
            cls.META_APP_ID,
            cls.META_APP_SECRET,
            cls.META_ACCESS_TOKEN,
            cls.META_AD_ACCOUNT_ID
        ]
        return all(var is not None for var in required_vars)


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
