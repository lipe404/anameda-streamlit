"""
Script para testar credenciais da API do Meta
"""
from config.settings import Config
import sys
import os

# Adiciona o diretório atual ao path ANTES dos imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


try:
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    FACEBOOK_SDK_AVAILABLE = True
except ImportError:
    print("⚠️ Facebook Business SDK não instalado. Execute: pip install facebook-business")
    FACEBOOK_SDK_AVAILABLE = False


def test_environment_variables():
    """Testa se as variáveis de ambiente estão carregadas"""
    print("🔍 Testando variáveis de ambiente...")

    Config.print_config()

    debug_info = Config.get_debug_info()
    print(f"\n📊 Informações de Debug:")
    for key, value in debug_info.items():
        print(f"   {key}: {value}")

    return Config.validate_credentials()


def test_facebook_api():
    """Testa a conexão com a API do Facebook"""
    if not FACEBOOK_SDK_AVAILABLE:
        return False

    print("\n🔗 Testando conexão com a API do Meta...")

    try:
        # Inicializa a API
        FacebookAdsApi.init(
            app_id=Config.META_APP_ID,
            app_secret=Config.META_APP_SECRET,
            access_token=Config.META_ACCESS_TOKEN
        )

        print("✅ API inicializada com sucesso")
        return True

    except Exception as e:
        print(f"❌ Erro ao inicializar API: {str(e)}")
        return False


def test_ad_account_access():
    """Testa acesso à conta de anúncios"""
    if not FACEBOOK_SDK_AVAILABLE:
        return False

    print("\n🎯 Testando acesso à conta de anúncios...")

    try:
        # Testa acesso à conta de anúncios
        ad_account = AdAccount(Config.META_AD_ACCOUNT_ID)
        account_info = ad_account.api_get(
            fields=['name', 'account_status', 'currency'])

        print(f"✅ Conta de anúncios acessada com sucesso:")
        print(f"   Nome: {account_info.get('name', 'N/A')}")
        print(f"   Status: {account_info.get('account_status', 'N/A')}")
        print(f"   Moeda: {account_info.get('currency', 'N/A')}")
        print(f"   ID: {Config.META_AD_ACCOUNT_ID}")

        return True

    except Exception as e:
        print(f"❌ Erro ao acessar conta de anúncios: {str(e)}")
        print(
            f"   Verifique se o ID da conta está correto: {Config.META_AD_ACCOUNT_ID}")
        return False


def test_campaigns_access():
    """Testa acesso às campanhas"""
    if not FACEBOOK_SDK_AVAILABLE:
        return False

    print("\n📊 Testando acesso às campanhas...")

    try:
        ad_account = AdAccount(Config.META_AD_ACCOUNT_ID)

        # Testa busca de campanhas
        campaigns = ad_account.get_campaigns(
            fields=['name', 'status', 'objective'],
            params={'limit': 10}
        )
        campaign_list = list(campaigns)

        print(f"✅ Encontradas {len(campaign_list)} campanhas")

        if campaign_list:
            print("   Campanhas encontradas:")
            for i, campaign in enumerate(campaign_list[:5]):
                name = campaign.get('name', 'N/A')
                status = campaign.get('status', 'N/A')
                objective = campaign.get('objective', 'N/A')
                print(f"   {i+1}. {name} - {status} ({objective})")
        else:
            print("   ⚠️ Nenhuma campanha encontrada na conta")

        return True

    except Exception as e:
        print(f"❌ Erro ao buscar campanhas: {str(e)}")
        return False


def test_insights_access():
    """Testa acesso aos insights"""
    if not FACEBOOK_SDK_AVAILABLE:
        return False

    print("\n📈 Testando acesso aos insights...")

    try:
        ad_account = AdAccount(Config.META_AD_ACCOUNT_ID)

        # Testa busca de insights
        insights = ad_account.get_insights(
            fields=['impressions', 'clicks', 'spend'],
            params={
                'date_preset': 'last_7d',
                'limit': 1
            }
        )
        insights_list = list(insights)

        if insights_list:
            insight = insights_list[0]
            print(f"✅ Insights acessados com sucesso:")
            print(f"   Impressões: {insight.get('impressions', 'N/A')}")
            print(f"   Clicks: {insight.get('clicks', 'N/A')}")
            print(f"   Gasto: R\$ {insight.get('spend', 'N/A')}")
        else:
            print(
                "   ⚠️ Nenhum insight encontrado (pode ser normal se não há campanhas ativas)")

        return True

    except Exception as e:
        print(f"❌ Erro ao acessar insights: {str(e)}")
        return False


def run_all_tests():
    """Executa todos os testes"""
    print("🚀 Iniciando testes completos da API do Meta...\n")

    tests_results = {
        'environment': test_environment_variables(),
        'api_connection': False,
        'ad_account': False,
        'campaigns': False,
        'insights': False
    }

    if tests_results['environment']:
        tests_results['api_connection'] = test_facebook_api()

        if tests_results['api_connection']:
            tests_results['ad_account'] = test_ad_account_access()

            if tests_results['ad_account']:
                tests_results['campaigns'] = test_campaigns_access()
                tests_results['insights'] = test_insights_access()

    # Resumo dos testes
    print("\n" + "="*50)
    print("📋 RESUMO DOS TESTES")
    print("="*50)

    for test_name, result in tests_results.items():
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    all_passed = all(tests_results.values())

    if all_passed:
        print("\n🎉 Todos os testes passaram! A API está funcionando corretamente.")
        print("   Você pode executar o aplicativo Streamlit com: streamlit run app.py")
    else:
        print("\n⚠️ Alguns testes falharam. Verifique as configurações acima.")

        if not tests_results['environment']:
            print("   • Verifique o arquivo .env e as variáveis de ambiente")
        elif not tests_results['api_connection']:
            print(
                "   • Verifique as credenciais da API (APP_ID, APP_SECRET, ACCESS_TOKEN)")
        elif not tests_results['ad_account']:
            print("   • Verifique o ID da conta de anúncios")
        else:
            print("   • Verifique as permissões da API")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
