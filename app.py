import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import pickle
import requests
import json
import time
from io import BytesIO
import base64

# Configuration de la page
st.set_page_config(
    page_title="🚀 Reddit Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un look professionnel
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    .signal-buy {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
    }
    .signal-sell {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%) !important;
    }
    .signal-hold {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%) !important;
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)

class TradingSystemCloud:
    """Système de trading adapté pour Streamlit Cloud"""
    
    def __init__(self):
        self.initialize_session_state()
        
    @st.cache_data
    def load_sample_models():
        """Charge des modèles d'exemple (simulation)"""
        # En production, vous chargeriez vos vrais modèles
        # Pour la démo, on simule la présence de modèles
        models = {
            'RandomForest_direction_1d': 'model_simulated',
            'XGBoost_direction_3d': 'model_simulated',
            'LinearRegression_return_1d': 'model_simulated'
        }
        return models
    
    def initialize_session_state(self):
        """Initialise les variables de session"""
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {
                'cash': 10000.0,
                'positions': {
                    'AAPL': {'quantity': 10, 'avg_price': 150.0},
                    'MSFT': {'quantity': 5, 'avg_price': 280.0},
                    'GOOGL': {'quantity': 2, 'avg_price': 2500.0}
                },
                'history': []
            }
        
        if 'predictions_history' not in st.session_state:
            st.session_state.predictions_history = []
        
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
    
    @st.cache_data(ttl=300)  # Cache pendant 5 minutes
    def get_stock_data(_self, ticker, period="1mo"):
        """Récupère les données boursières avec cache"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            st.error(f"Erreur lors de la récupération des données pour {ticker}: {e}")
            return pd.DataFrame()
    
    def simulate_sentiment_data(self, ticker):
        """Simule des données de sentiment pour la démo"""
        # Dans un vrai déploiement, ceci viendrait de votre API Reddit
        return {
            'sentiment_avg': np.random.uniform(-0.3, 0.3),
            'sentiment_ma3': np.random.uniform(-0.2, 0.2),
            'sentiment_ma7': np.random.uniform(-0.1, 0.1),
            'post_count': np.random.randint(50, 200),
            'confidence': np.random.uniform(0.6, 0.9),
            'buzz_score': np.random.uniform(0.1, 0.8)
        }
    
    def predict_ticker(self, ticker, sentiment_data):
        """Fait une prédiction pour un ticker (simulation)"""
        # Simulation d'une prédiction ML
        base_prob = 0.5 + sentiment_data['sentiment_avg'] * 0.5
        noise = np.random.uniform(-0.1, 0.1)
        probability = np.clip(base_prob + noise, 0.1, 0.9)
        
        direction = "UP" if probability > 0.5 else "DOWN"
        confidence = abs(probability - 0.5) * 2
        expected_return = sentiment_data['sentiment_avg'] * 0.05 + np.random.uniform(-0.02, 0.02)
        
        return {
            'direction': direction,
            'probability': probability,
            'confidence': confidence,
            'expected_return': expected_return,
            'model_used': 'RandomForest_direction_1d'
        }
    
    def generate_trading_signal(self, prediction, confidence_threshold=0.6):
        """Génère un signal de trading"""
        confidence = prediction['confidence']
        direction = prediction['direction']
        
        if confidence > confidence_threshold:
            signal = "BUY" if direction == "UP" else "SELL"
            confidence_level = "HIGH" if confidence > 0.8 else "MEDIUM"
        else:
            signal = "HOLD"
            confidence_level = "LOW"
        
        return {
            'signal': signal,
            'confidence_level': confidence_level,
            'reason': f"Prediction {direction} with {confidence:.1%} confidence",
            'timestamp': datetime.now()
        }
    
    def calculate_portfolio_value(self, current_prices):
        """Calcule la valeur actuelle du portefeuille"""
        total_value = st.session_state.portfolio['cash']
        
        for ticker, position in st.session_state.portfolio['positions'].items():
            if ticker in current_prices:
                total_value += current_prices[ticker] * position['quantity']
        
        return total_value
    
    def add_alert(self, message, level="INFO"):
        """Ajoute une alerte"""
        alert = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message
        }
        st.session_state.alerts.append(alert)
        
        # Garder seulement les 50 dernières alertes
        if len(st.session_state.alerts) > 50:
            st.session_state.alerts = st.session_state.alerts[-50:]

# Initialiser le système
@st.cache_resource
def get_trading_system():
    return TradingSystemCloud()

system = get_trading_system()

# Interface utilisateur principale
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Reddit Trading Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Système de trading algorithmique basé sur l\'analyse de sentiment Reddit</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Sélection du ticker
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "CRM", "UBER"]
    selected_ticker = st.sidebar.selectbox("📈 Sélectionner un ticker", tickers)
    
    # Paramètres
    confidence_threshold = st.sidebar.slider("🎯 Seuil de confiance", 0.5, 0.9, 0.6, 0.05)
    auto_refresh = st.sidebar.checkbox("🔄 Actualisation automatique (30s)")
    
    # Informations système
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Statut Système")
    st.sidebar.success("🟢 Système opérationnel")
    st.sidebar.info(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    # Simuler des données de portfolio
    current_prices = {}
    for ticker in st.session_state.portfolio['positions'].keys():
        stock_data = system.get_stock_data(ticker, "1d")
        if not stock_data.empty:
            current_prices[ticker] = stock_data['Close'].iloc[-1]
    
    portfolio_value = system.calculate_portfolio_value(current_prices)
    daily_change = np.random.uniform(-500, 800)  # Simulation
    
    with col1:
        st.metric(
            "💰 Valeur Portefeuille", 
            f"${portfolio_value:,.0f}",
            f"{daily_change:+.0f}"
        )
    
    with col2:
        positions_count = len(st.session_state.portfolio['positions'])
        st.metric("📊 Positions Actives", positions_count, "+1")
    
    with col3:
        accuracy = np.random.uniform(65, 75)
        st.metric("🎯 Accuracy Modèle", f"{accuracy:.1f}%", "+2.1%")
    
    with col4:
        signals_today = np.random.randint(8, 20)
        st.metric("⚡ Signaux Aujourd'hui", signals_today, "+3")
    
    # Section principale
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader(f"📈 Analyse de {selected_ticker}")
        
        # Récupérer et afficher les données de prix
        stock_data = system.get_stock_data(selected_ticker)
        
        if not stock_data.empty:
            # Graphique des prix
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name=selected_ticker
            ))
            
            fig.update_layout(
                title=f"Prix de {selected_ticker} (30 derniers jours)",
                yaxis_title="Prix ($)",
                xaxis_title="Date",
                height=400,
                showlegend=False,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prédiction et signal
            st.subheader("🔮 Prédiction du Modèle")
            
            # Bouton pour générer une nouvelle prédiction
            if st.button("🎲 Nouvelle Prédiction", key="predict_btn"):
                sentiment_data = system.simulate_sentiment_data(selected_ticker)
                prediction = system.predict_ticker(selected_ticker, sentiment_data)
                signal = system.generate_trading_signal(prediction, confidence_threshold)
                
                # Sauvegarder dans l'historique
                st.session_state.predictions_history.append({
                    'ticker': selected_ticker,
                    'prediction': prediction,
                    'signal': signal,
                    'sentiment': sentiment_data
                })
                
                system.add_alert(
                    f"Nouvelle prédiction pour {selected_ticker}: {signal['signal']}", 
                    "INFO"
                )
            
            # Afficher la dernière prédiction
            if st.session_state.predictions_history:
                last_pred = st.session_state.predictions_history[-1]
                
                if last_pred['ticker'] == selected_ticker:
                    pred = last_pred['prediction']
                    signal = last_pred['signal']
                    
                    # Métriques de prédiction
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    
                    with pred_col1:
                        direction_emoji = "🟢" if pred['direction'] == "UP" else "🔴"
                        st.metric("Direction", f"{direction_emoji} {pred['direction']}")
                    
                    with pred_col2:
                        conf_color = "🟢" if pred['confidence'] > 0.7 else "🟡" if pred['confidence'] > 0.6 else "🔴"
                        st.metric("Confiance", f"{conf_color} {pred['confidence']:.1%}")
                    
                    with pred_col3:
                        return_color = "🟢" if pred['expected_return'] > 0 else "🔴"
                        st.metric("Retour Attendu", f"{return_color} {pred['expected_return']:+.2%}")
                    
                    # Signal de trading
                    signal_class = f"signal-{signal['signal'].lower()}"
                    
                    st.markdown(f"""
                    <div class="prediction-box {signal_class}">
                        <h3>📢 Signal de Trading</h3>
                        <h2>{signal['signal']}</h2>
                        <p>Niveau de confiance: {signal['confidence_level']}</p>
                        <p>{signal['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Détails sentiment
                    with st.expander("📊 Détails du Sentiment"):
                        sentiment = last_pred['sentiment']
                        sent_col1, sent_col2 = st.columns(2)
                        
                        with sent_col1:
                            st.metric("Sentiment Moyen", f"{sentiment['sentiment_avg']:+.3f}")
                            st.metric("Posts Analysés", f"{sentiment['post_count']}")
                        
                        with sent_col2:
                            st.metric("Buzz Score", f"{sentiment['buzz_score']:.3f}")
                            st.metric("Confiance", f"{sentiment['confidence']:.1%}")
            
            else:
                st.info("Cliquez sur 'Nouvelle Prédiction' pour analyser ce ticker")
        
        else:
            st.error(f"Impossible de récupérer les données pour {selected_ticker}")
    
    with col_right:
        st.subheader("📊 Portefeuille")
        
        # Positions actuelles
        positions_data = []
        total_value = 0
        
        for ticker, position in st.session_state.portfolio['positions'].items():
            current_price = current_prices.get(ticker, position['avg_price'])
            market_value = current_price * position['quantity']
            pnl = market_value - (position['avg_price'] * position['quantity'])
            
            positions_data.append({
                'Ticker': ticker,
                'Quantité': position['quantity'],
                'Prix Moyen': f"${position['avg_price']:.2f}",
                'Prix Actuel': f"${current_price:.2f}",
                'Valeur': f"${market_value:.0f}",
                'P&L': f"${pnl:+.0f}"
            })
            total_value += market_value
        
        portfolio_df = pd.DataFrame(positions_data)
        st.dataframe(portfolio_df, use_container_width=True)
        
        # Graphique de répartition
        if positions_data:
            values = [float(pos['Valeur'].replace('$', '').replace(',', '')) for pos in positions_data]
            tickers = [pos['Ticker'] for pos in positions_data]
            
            fig_pie = px.pie(
                values=values, 
                names=tickers,
                title="Répartition du Portefeuille"
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Alertes récentes
        st.subheader("🚨 Alertes Récentes")
        
        if st.session_state.alerts:
            for alert in st.session_state.alerts[-5:]:  # 5 dernières alertes
                alert_time = alert['timestamp'].strftime("%H:%M")
                if alert['level'] == 'INFO':
                    st.info(f"ℹ️ {alert_time} - {alert['message']}")
                elif alert['level'] == 'WARNING':
                    st.warning(f"⚠️ {alert_time} - {alert['message']}")
                else:
                    st.success(f"✅ {alert_time} - {alert['message']}")
        else:
            st.info("Aucune alerte récente")
    
    # Section des graphiques avancés
    st.markdown("---")
    st.subheader("📈 Analyses Avancées")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**📊 Performance Historique**")
        
        # Simuler des données de performance
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        base_value = 10000
        returns = np.random.normal(0.001, 0.02, 30)  # 0.1% return moyen, 2% volatilité
        portfolio_values = [base_value]
        
        for ret in returns[1:]:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        perf_df = pd.DataFrame({
            'Date': dates,
            'Valeur': portfolio_values
        })
        
        fig_perf = px.line(
            perf_df, 
            x='Date', 
            y='Valeur',
            title="Évolution du Portefeuille (30j)"
        )
        fig_perf.update_layout(height=300)
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with chart_col2:
        st.markdown("**⚡ Signaux de Trading**")
        
        # Graphique des signaux
        hours = [f"{i:02d}:00" for i in range(9, 17)]  # Heures de marché
        buy_signals = np.random.poisson(2, len(hours))
        sell_signals = np.random.poisson(1.5, len(hours))
        hold_signals = np.random.poisson(3, len(hours))
        
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Bar(name='BUY', x=hours, y=buy_signals, marker_color='green'))
        fig_signals.add_trace(go.Bar(name='SELL', x=hours, y=sell_signals, marker_color='red'))
        fig_signals.add_trace(go.Bar(name='HOLD', x=hours, y=hold_signals, marker_color='orange'))
        
        fig_signals.update_layout(
            title="Signaux par Heure",
            barmode='stack',
            height=300
        )
        st.plotly_chart(fig_signals, use_container_width=True)
    
    # Section d'historique des prédictions
    if st.session_state.predictions_history:
        st.markdown("---")
        st.subheader("📋 Historique des Prédictions")
        
        # Créer un DataFrame à partir de l'historique
        history_data = []
        for entry in st.session_state.predictions_history[-10:]:  # 10 dernières
            history_data.append({
                'Heure': entry['signal']['timestamp'].strftime('%H:%M:%S'),
                'Ticker': entry['ticker'],
                'Direction': entry['prediction']['direction'],
                'Confiance': f"{entry['prediction']['confidence']:.1%}",
                'Signal': entry['signal']['signal'],
                'Retour Attendu': f"{entry['prediction']['expected_return']:+.2%}"
            })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()

# Section d'aide et informations
def show_help():
    st.markdown("---")
    st.subheader("ℹ️ Guide d'Utilisation")
    
    with st.expander("🚀 Comment utiliser ce système"):
        st.markdown("""
        ### 📈 Analyse des Tickers
        1. **Sélectionnez un ticker** dans la sidebar
        2. **Cliquez sur 'Nouvelle Prédiction'** pour analyser
        3. **Observez le signal** généré (BUY/SELL/HOLD)
        4. **Ajustez le seuil de confiance** selon vos préférences
        
        ### 📊 Interprétation des Signaux
        - **🟢 BUY**: Le modèle prédit une hausse probable
        - **🔴 SELL**: Le modèle prédit une baisse probable  
        - **🟡 HOLD**: Confiance insuffisante ou signal neutre
        
        ### ⚙️ Paramètres
        - **Seuil de confiance**: Plus élevé = signaux plus rares mais plus fiables
        - **Auto-refresh**: Actualisation automatique toutes les 30 secondes
        
        ### 📱 Fonctionnalités
        - **Données temps réel** via Yahoo Finance
        - **Prédictions ML** basées sur l'analyse de sentiment
        - **Suivi du portefeuille** virtuel
        - **Historique** des prédictions
        """)
    
    with st.expander("⚠️ Avertissements Importants"):
        st.warning("""
        **🚨 UTILISATION ÉDUCATIVE UNIQUEMENT**
        
        - Ce système est conçu à des **fins éducatives et de démonstration**
        - **Ne pas utiliser** pour des décisions d'investissement réelles
        - Les prédictions sont basées sur des **données simulées**
        - **Toujours consulter** un conseiller financier professionnel
        - Les performances passées ne garantissent pas les résultats futurs
        """)

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h4>🤖 Reddit Trading System v1.0</h4>
        <p>Développé avec ❤️ en utilisant Streamlit et Machine Learning</p>
        <p><em>Système de trading algorithmique basé sur l'analyse de sentiment Reddit</em></p>
    </div>
    """, unsafe_allow_html=True)

# Exécution principale
if __name__ == "__main__":
    main()
    show_help()
    show_footer()
