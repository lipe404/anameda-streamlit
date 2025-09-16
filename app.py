"""
Aplicativo principal Streamlit para Meta Ads Analytics Dashboard
"""
from src.utils.visualizations import VisualizationUtils
from src.utils.data_processor import DataProcessor
from src.analytics.prescriptive import PrescriptiveAnalytics
from src.analytics.predictive import PredictiveAnalytics
from src.analytics.diagnostic import DiagnosticAnalytics
from src.analytics.descriptive import DescriptiveAnalytics
from src.api.meta_api import MetaAdsAPI
from config.settings import Config
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importações dos módulos

# Configuração da página
st.set_page_config(
    page_title="AnaMeDa - Analytics Meta Dados",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Inicializa variáveis de estado da sessão"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'campaign_data' not in st.session_state:
        st.session_state.campaign_data = pd.DataFrame()
    if 'use_sample_data' not in st.session_state:
        st.session_state.use_sample_data = False


def load_data():
    """Carrega dados das campanhas"""
    try:
        if st.session_state.use_sample_data:
            # Usa dados de exemplo
            data = DataProcessor.create_sample_data(days=60)
            st.session_state.campaign_data = data
            st.session_state.data_loaded = True
            return data
        else:
            # Verifica credenciais
            if not Config.validate_credentials():
                st.error(
                    "⚠️ Credenciais da API do Meta não configuradas. Usando dados de exemplo.")
                data = DataProcessor.create_sample_data(days=60)
                st.session_state.campaign_data = data
                st.session_state.data_loaded = True
                return data

            # Conecta com a API do Meta
            with st.spinner("Conectando com a API do Meta Business Suite..."):
                meta_api = MetaAdsAPI()
                data = meta_api.get_all_campaigns_data()

                if data.empty:
                    st.warning(
                        "Nenhum dado encontrado. Usando dados de exemplo.")
                    data = DataProcessor.create_sample_data(days=60)

                # Processa e limpa os dados
                data = DataProcessor.clean_campaign_data(data)
                data = DataProcessor.calculate_derived_metrics(data)

                st.session_state.campaign_data = data
                st.session_state.data_loaded = True
                return data

    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        st.info("Carregando dados de exemplo...")
        data = DataProcessor.create_sample_data(days=60)
        st.session_state.campaign_data = data
        st.session_state.data_loaded = True
        return data


def show_overview_page():
    """Página de visão geral"""
    st.markdown('<h1 class="main-header">📊 AnaMeDa - Analytics Meta Dados</h1>',
                unsafe_allow_html=True)

    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")

        # Opção para usar dados de exemplo
        use_sample = st.checkbox(
            "Usar dados de exemplo",
            value=st.session_state.use_sample_data,
            help="Marque para usar dados simulados em vez da API do Meta"
        )

        if use_sample != st.session_state.use_sample_data:
            st.session_state.use_sample_data = use_sample
            st.session_state.data_loaded = False

        # Botão para recarregar dados
        if st.button("🔄 Recarregar Dados"):
            st.session_state.data_loaded = False
            st.experimental_rerun()

        # Status da conexão
        st.subheader("📡 Status da Conexão")
        if Config.validate_credentials() and not st.session_state.use_sample_data:
            st.success("✅ API Configurada")
        else:
            st.warning("⚠️ Usando Dados de Exemplo")

    # Carrega dados se necessário
    if not st.session_state.data_loaded:
        data = load_data()
    else:
        data = st.session_state.campaign_data

    if data.empty:
        st.error("❌ Nenhum dado disponível")
        return

    # Análise descritiva
    descriptive = DescriptiveAnalytics(data)
    overview = descriptive.get_campaign_performance_overview()

    # KPIs principais
    st.subheader("📈 Métricas Principais")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 Gasto Total",
            value=f"R$ {overview.get('total_spend', 0):,.2f}",
            delta="5.2%"
        )

    with col2:
        st.metric(
            label="👁️ Impressões",
            value=f"{overview.get('total_impressions', 0):,}",
            delta="12.3%"
        )

    with col3:
        st.metric(
            label="🖱️ Clicks",
            value=f"{overview.get('total_clicks', 0):,}",
            delta="8.7%"
        )

    with col4:
        st.metric(
            label="📊 CTR Médio",
            value=f"{overview.get('avg_ctr', 0):.2f}%",
            delta="2.1%"
        )

    # Informações do período
    st.info(f"📅 **Período:** {overview.get('date_range', 'N/A')} | "
            f"🎯 **Campanhas Ativas:** {overview.get('campaigns_count', 0)}")

    # Gráficos principais
    st.subheader("📊 Análise Visual")

    charts = descriptive.create_performance_charts()

    # Layout em duas colunas
    col1, col2 = st.columns(2)

    with col1:
        if 'spend_timeline' in charts:
            st.plotly_chart(charts['spend_timeline'], use_container_width=True)

        if 'ctr_by_campaign' in charts:
            st.plotly_chart(charts['ctr_by_campaign'],
                            use_container_width=True)

    with col2:
        if 'impressions_clicks' in charts:
            st.plotly_chart(charts['impressions_clicks'],
                            use_container_width=True)

        if 'spend_distribution' in charts:
            st.plotly_chart(charts['spend_distribution'],
                            use_container_width=True)

    # Top campanhas
    st.subheader("🏆 Top Campanhas por CTR")
    top_campaigns = descriptive.get_top_performing_campaigns('ctr', 5)

    if not top_campaigns.empty:
        st.dataframe(
            top_campaigns[['campaign_name', 'spend',
                           'impressions', 'clicks', 'ctr', 'cpc']].round(2),
            use_container_width=True
        )


def show_diagnostic_page():
    """Página de análise diagnóstica"""
    st.header("🔍 Análise Diagnóstica")

    data = st.session_state.campaign_data

    if data.empty:
        st.warning("⚠️ Carregue os dados primeiro na página Overview")
        return

    diagnostic = DiagnosticAnalytics(data)

    # Análise de tendências
    st.subheader("📈 Análise de Tendências")
    trends = diagnostic.analyze_performance_trends()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Tendências Identificadas:**")
        for metric, trend_data in trends.items():
            trend_icon = "📈" if trend_data['trend'] == 'crescente' else "📉" if trend_data['trend'] == 'decrescente' else "➡️"
            significance = "✅" if trend_data['significance'] == 'significativo' else "❌"

            st.write(
                f"{trend_icon} **{metric.upper()}**: {trend_data['trend']} {significance}")

    with col2:
        # Gráfico de correlação
        correlation_matrix = diagnostic.correlation_analysis()
        if not correlation_matrix.empty:
            fig_corr = VisualizationUtils.create_correlation_heatmap(data)
            st.plotly_chart(fig_corr, use_container_width=True)

    # Detecção de anomalias
    st.subheader("🚨 Detecção de Anomalias")

    metric_for_anomaly = st.selectbox(
        "Selecione a métrica para análise de anomalias:",
        ['spend', 'impressions', 'clicks', 'ctr', 'cpc']
    )

    anomaly_data = diagnostic.identify_anomalies(metric_for_anomaly)
    anomalies_count = anomaly_data['is_anomaly'].sum()

    col1, col2 = st.columns([1, 3])

    with col1:
        st.metric("🚨 Anomalias Detectadas", anomalies_count)

        if anomalies_count > 0:
            st.write("**Datas com Anomalias:**")
            anomaly_dates = anomaly_data[anomaly_data['is_anomaly']]['date'].dt.strftime(
                '%d/%m/%Y').tolist()
            for date in anomaly_dates[:5]:  # Mostra apenas as 5 primeiras
                st.write(f"• {date}")

    with col2:
        # Gráfico de anomalias
        fig_anomaly = px.scatter(
            anomaly_data,
            x='date',
            y=metric_for_anomaly,
            color='is_anomaly',
            title=f'Detecção de Anomalias - {metric_for_anomaly.title()}',
            color_discrete_map={True: 'red', False: 'blue'}
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)

    # Análise de eficiência
    st.subheader("⚡ Análise de Eficiência das Campanhas")

    efficiency_data = diagnostic.campaign_efficiency_analysis()

    if not efficiency_data.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Distribuição por quartil de eficiência
            quartile_counts = efficiency_data['efficiency_quartile'].value_counts(
            )
            fig_quartile = px.pie(
                values=quartile_counts.values,
                names=quartile_counts.index,
                title="Distribuição de Campanhas por Eficiência"
            )
            st.plotly_chart(fig_quartile, use_container_width=True)

        with col2:
            # Scatter plot eficiência
            fig_efficiency = px.scatter(
                efficiency_data,
                x='cpc',
                y='ctr',
                size='spend',
                color='efficiency_quartile',
                title='Análise de Eficiência: CTR vs CPC',
                hover_data=['campaign_name']
            )
            st.plotly_chart(fig_efficiency, use_container_width=True)

        # Tabela de eficiência
        st.write("**Ranking de Eficiência das Campanhas:**")
        efficiency_display = efficiency_data[[
            'campaign_name', 'spend', 'ctr', 'cpc', 'efficiency_score', 'efficiency_quartile']].round(2)
        efficiency_display = efficiency_display.sort_values(
            'efficiency_score', ascending=False)
        st.dataframe(efficiency_display, use_container_width=True)


def show_predictive_page():
    """Página de análise preditiva"""
    st.header("🔮 Análise Preditiva")

    data = st.session_state.campaign_data

    if data.empty:
        st.warning("⚠️ Carregue os dados primeiro na página Overview")
        return

    # Configurações de previsão
    col1, col2, col3 = st.columns(3)

    with col1:
        target_metric = st.selectbox(
            "Métrica para Previsão:",
            ['spend', 'impressions', 'clicks', 'ctr'],
            help="Selecione a métrica que deseja prever"
        )

    with col2:
        prediction_days = st.slider(
            "Dias para Prever:",
            min_value=3,
            max_value=30,
            value=7,
            help="Número de dias futuros para previsão"
        )

    with col3:
        models_to_use = st.multiselect(
            "Modelos para Usar:",
            ['ARIMA', 'TensorFlow', 'Linear Regression'],
            default=['ARIMA', 'Linear Regression'],
            help="Selecione os modelos para treinamento"
        )

    if st.button("🚀 Treinar Modelos e Gerar Previsões"):
        predictive = PredictiveAnalytics(data)

        with st.spinner("Treinando modelos... Isso pode levar alguns minutos."):
            # Treina modelos selecionados
            training_results = {}

            if 'ARIMA' in models_to_use:
                with st.spinner("Treinando modelo ARIMA..."):
                    training_results['arima'] = predictive.train_arima_model(
                        target_metric)

            if 'TensorFlow' in models_to_use:
                with st.spinner("Treinando modelo TensorFlow..."):
                    training_results['tensorflow'] = predictive.train_tensorflow_model(
                        target_metric, epochs=30)

            if 'Linear Regression' in models_to_use:
                with st.spinner("Treinando modelo de Regressão Linear..."):
                    training_results['linear_regression'] = predictive.train_linear_regression_model(
                        target_metric)

        # Mostra resultados do treinamento
        st.subheader("📊 Resultados do Treinamento")

        for model_name, result in training_results.items():
            if result.get('success', False):
                st.success(
                    f"✅ {model_name.upper()}: Modelo treinado com sucesso")
            else:
                st.error(
                    f"❌ {model_name.upper()}: {result.get('error', 'Erro desconhecido')}")

        # Gera previsões
        st.subheader("🔮 Previsões Geradas")

        predictions = predictive.generate_predictions(
            target_metric, prediction_days)
        ensemble = predictive.create_ensemble_prediction(
            target_metric, prediction_days)

        # Gráfico de previsões
        charts = predictive.create_prediction_charts(
            target_metric, prediction_days)

        if 'predictions_comparison' in charts:
            st.plotly_chart(
                charts['predictions_comparison'], use_container_width=True)

        # Tabela de previsões
        if not ensemble.empty:
            st.subheader("📋 Tabela de Previsões")

            # Formata tabela
            ensemble_display = ensemble.copy()
            ensemble_display['date'] = ensemble_display['date'].dt.strftime(
                '%d/%m/%Y')
            ensemble_display['ensemble_prediction'] = ensemble_display['ensemble_prediction'].round(
                2)

            st.dataframe(ensemble_display, use_container_width=True)

            # Estatísticas das previsões
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📈 Previsão Média",
                    f"{ensemble['ensemble_prediction'].mean():.2f}"
                )

            with col2:
                st.metric(
                    "Variação Total",
                    f"{ensemble['ensemble_prediction'].std():.2f}"
                )

            with col3:
                trend = "↗️ Crescente" if ensemble['ensemble_prediction'].iloc[-1] > ensemble['ensemble_prediction'].iloc[0] else "↘️ Decrescente"
                st.metric("📈 Tendência", trend)

        # Performance dos modelos
        st.subheader("🎯 Performance dos Modelos")
        model_performance = predictive.get_model_performance_summary(
            target_metric)

        for model_name, performance in model_performance.items():
            with st.expander(f"📊 {model_name.upper()} - Detalhes"):
                st.json(performance)


def show_prescriptive_page():
    """Página de análise prescritiva"""
    st.header("💡 Análise Prescritiva e Recomendações")

    data = st.session_state.campaign_data

    if data.empty:
        st.warning("⚠️ Carregue os dados primeiro na página Overview")
        return

    prescriptive = PrescriptiveAnalytics(data)

    # Plano de ação geral
    st.subheader("🎯 Plano de Ação Estratégico")

    action_plan = prescriptive.generate_action_plan()

    # Ações imediatas
    st.markdown("### 🚨 Ações Imediatas (0-3 dias)")
    for action in action_plan['immediate_actions']:
        st.write(f"• {action}")

    # Ações de curto prazo
    st.markdown("### ⏱️ Ações de Curto Prazo (1-2 semanas)")
    for action in action_plan['short_term_actions']:
        st.write(f"• {action}")

    # Ações de longo prazo
    st.markdown("### 📅 Ações de Longo Prazo (1+ mês)")
    for action in action_plan['long_term_actions']:
        st.write(f"• {action}")

    # Recomendações específicas
    st.subheader("🎯 Recomendações Específicas")

    # Tabs para diferentes tipos de recomendações
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💰 Orçamento", "🎯 Lances", "👥 Targeting", "🎨 Criativos"])

    with tab1:
        st.markdown("#### 💰 Otimização de Orçamento")
        budget_recs = prescriptive.budget_optimization_recommendations()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📈 Aumentar Orçamento:**")
            for campaign in budget_recs['increase_budget']['campaigns']:
                st.write(f"• {campaign}")
            st.info(f"💡 {budget_recs['increase_budget']['reason']}")

        with col2:
            st.markdown("**📉 Reduzir Orçamento:**")
            for campaign in budget_recs['decrease_budget']['campaigns']:
                st.write(f"• {campaign}")
            st.warning(f"⚠️ {budget_recs['decrease_budget']['reason']}")

        # Potencial economia
        potential_savings = budget_recs['total_budget_reallocation']['potential_savings']
        st.success(f"💰 **Economia Potencial:** R$ {potential_savings:,.2f}")

    with tab2:
        st.markdown("#### 🎯 Estratégia de Lances")
        bidding_recs = prescriptive.bidding_strategy_recommendations()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📉 Reduzir Lances:**")
            for campaign in bidding_recs['reduce_bids']['campaigns']:
                st.write(f"• {campaign}")
            st.info(f"💡 {bidding_recs['reduce_bids']['action']}")

        with col2:
            st.markdown("**📈 Aumentar Lances:**")
            for campaign in bidding_recs['increase_bids']['campaigns']:
                st.write(f"• {campaign}")
            st.success(f"🚀 {bidding_recs['increase_bids']['action']}")

    with tab3:
        st.markdown("#### 👥 Otimização de Targeting")
        targeting_recs = prescriptive.targeting_optimization_recommendations()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📅 Otimização de Cronograma:**")
            st.write("**Melhores dias:**")
            for day in targeting_recs['schedule_optimization']['increase_budget_days']:
                st.write(f"• {day}")

        with col2:
            st.markdown("**👥 Expansão de Audiência:**")
            for campaign in targeting_recs['audience_expansion']['high_performing_campaigns'][:3]:
                st.write(f"• {campaign}")
            st.info("💡 Expandir audiências similares")

    with tab4:
        st.markdown("#### 🎨 Otimização de Criativos")
        creative_recs = prescriptive.creative_optimization_recommendations()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔄 Renovar Criativos:**")
            for campaign in creative_recs['creative_refresh']['campaigns']:
                st.write(f"• {campaign}")
            st.warning("⚠️ Fadiga de criativo detectada")

        with col2:
            st.markdown("**🧪 Testar Variações:**")
            for campaign in creative_recs['creative_testing']['campaigns'][:3]:
                st.write(f"• {campaign}")
            st.success("🚀 Alto potencial para testes")

    # Clustering de campanhas
    st.subheader("🎯 Segmentação de Campanhas")

    clustering_insights = prescriptive.campaign_clustering_insights()

    for cluster_name, cluster_data in clustering_insights.items():
        with st.expander(f"📊 {cluster_name.replace('_', ' ').title()} ({cluster_data['characteristics']['campaign_count']} campanhas)"):

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Características:**")
                chars = cluster_data['characteristics']
                st.write(f"• Gasto médio: R$ {chars['avg_spend']:,.2f}")
                st.write(f"• CTR médio: {chars['avg_ctr']:.2f}%")
                st.write(f"• CPC médio: R$ {chars['avg_cpc']:.2f}")

            with col2:
                st.markdown("**Campanhas:**")
                for campaign in cluster_data['campaigns'][:5]:
                    st.write(f"• {campaign}")

            st.markdown("**Recomendações:**")
            for rec in cluster_data['recommendations']:
                st.write(f"💡 {rec}")


def main():
    """Função principal do aplicativo"""

    # Inicializa estado da sessão
    initialize_session_state()

    # Menu lateral
    st.sidebar.title("🚀 Meta Ads Analytics")
    st.sidebar.markdown("---")

    # Navegação
    pages = {
        "Overview": show_overview_page,
        "Análise Diagnóstica": show_diagnostic_page,
        "Análise Preditiva": show_predictive_page,
        "Recomendações": show_prescriptive_page
    }

    selected_page = st.sidebar.selectbox(
        "Selecione a página:", list(pages.keys()))

    # Informações da API
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Configuração da API")

    if Config.validate_credentials():
        st.sidebar.success("✅ Credenciais configuradas")
    else:
        st.sidebar.error("❌ Configure as credenciais")
        st.sidebar.markdown("""
        **Para configurar a API do Meta:**
        1. Crie um arquivo `.env`
        2. Adicione suas credenciais:
        ```
        META_APP_ID=seu_app_id
        META_APP_SECRET=seu_app_secret
        META_ACCESS_TOKEN=seu_token
        META_AD_ACCOUNT_ID=act_sua_conta
        ```
        """)

    # Executa página selecionada
    pages[selected_page]()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center'>
        <small>
        🚀 Meta Ads Analytics Dashboard<br>
        Desenvolvido com Streamlit<br>
        © 2025
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
