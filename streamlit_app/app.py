"""Dashboard de monitorización semanal del Sistema de Alerta Temprana (EWS) de Estrés
Financiero Sistémico. El modelo (LightGBM) y la configuración se entrenan y exportan desde
notebooks/01_datos_eda_feature_engineering.ipynb (Sección 9) — este script solo hace inferencia
sobre datos frescos de FRED, sin reentrenar nada.
"""

import json
import os
from datetime import date

import lightgbm as lgb
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from fredapi import Fred

st.set_page_config(page_title='EWS Estrés Financiero', page_icon='🚦', layout='wide')

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')

TREND_VARS = {
    'baa10y': 'Spread corporativo Baa (BAA10Y)',
    'vix': 'Volatilidad implícita (VIX)',
    'unrate': 'Tasa de desempleo (%)',
    'cpi_yoy': 'Inflación interanual (%)',
}


@st.cache_resource
def load_artifacts():
    # Formato nativo de LightGBM (Booster), no un .pkl del wrapper de scikit-learn: es portable
    # entre versiones de lightgbm/sklearn/Python distintas a las de Colab, donde se entrenó.
    model = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, 'lgbm_model.txt'))
    with open(os.path.join(ARTIFACTS_DIR, 'config.json')) as f:
        config = json.load(f)
    backtest_path = os.path.join(ARTIFACTS_DIR, 'backtest_summary.json')
    backtest_summary = pd.read_json(backtest_path) if os.path.exists(backtest_path) else None
    return model, config, backtest_summary


def to_weekly(series: pd.Series, max_ffill_days: int, all_days: pd.DatetimeIndex,
              weekly_index: pd.DatetimeIndex) -> pd.Series:
    """Misma lógica que en el notebook (Sección 3): ffill a diario con límite de días, muestreo semanal."""
    s = series.dropna().sort_index()
    daily = s.reindex(s.index.union(all_days)).ffill(limit=max_ffill_days).reindex(all_days)
    return daily.reindex(weekly_index)


@st.cache_data(ttl=3600, show_spinner='Descargando datos de FRED y recalculando la señal...')
def fetch_and_build_features(api_key, start_date, series_config, lags, rolling_windows):
    fred = Fred(api_key=api_key)
    end_date = date.today().isoformat()

    raw_series = {
        series_id: fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        for series_id in series_config
    }

    weekly_index = pd.date_range(start_date, end_date, freq='W-FRI')
    all_days = pd.date_range(start_date, end_date, freq='D')

    weekly_cols = {
        meta['name']: to_weekly(raw_series[series_id], meta['max_ffill_days'], all_days, weekly_index)
        for series_id, meta in series_config.items()
    }
    df_weekly = pd.DataFrame(weekly_cols, index=weekly_index)
    df_weekly.index.name = 'date'

    cpi_yoy = raw_series['CPIAUCSL'].pct_change(12) * 100
    df_weekly['cpi_yoy'] = to_weekly(cpi_yoy, series_config['CPIAUCSL']['max_ffill_days'], all_days, weekly_index)
    df_weekly = df_weekly.drop(columns=['cpi']).dropna(subset=['nfci'])

    all_predictor_cols = [c for c in df_weekly.columns if c != 'nfci'] + ['nfci']
    df_features = df_weekly.copy()
    for col in all_predictor_cols:
        for lag in lags:
            df_features[f'{col}_lag{lag}'] = df_weekly[col].shift(lag)
        shifted = df_weekly[col].shift(1)
        for window in rolling_windows:
            df_features[f'{col}_roll{window}_mean'] = shifted.rolling(window).mean()
            df_features[f'{col}_roll{window}_std'] = shifted.rolling(window).std()

    return df_weekly, df_features


def get_api_key():
    if 'FRED_API_KEY' in st.secrets:
        return st.secrets['FRED_API_KEY']
    return st.sidebar.text_input(
        'FRED API Key', type='password',
        help='Gratuita en https://fred.stlouisfed.org/docs/api/api_key.html',
    )


def render_driver_chart(model, latest_row, feature_cols, top_n=8):
    """Contribución de cada variable a la predicción actual, vía Booster.predict(pred_contrib=True)
    — es la misma descomposición aditiva de SHAP (TreeSHAP), nativa de LightGBM, sin depender de
    la librería `shap` completa (más pesada y con más riesgo de fallo en el build de despliegue)."""
    contrib = model.predict(latest_row[feature_cols], pred_contrib=True)[0]
    contrib_series = pd.Series(contrib[:-1], index=feature_cols)  # última columna = valor base
    top = contrib_series.reindex(contrib_series.abs().sort_values(ascending=False).index).head(top_n)
    top = top.sort_values()

    colors = ['#d62728' if v > 0 else '#2ca02c' for v in top.values]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation='h', marker_color=colors))
    fig.update_layout(
        title=f'Qué variables empujan la predicción de esta semana (top {top_n})',
        xaxis_title='Contribución al NFCI predicho (rojo = sube el estrés, verde = lo baja)',
        height=340, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def render_trend_panel(df_weekly, weeks=104):
    recent = df_weekly.tail(weeks)
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(TREND_VARS.values()))
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (col, _), (row, pos) in zip(TREND_VARS.items(), positions):
        fig.add_trace(
            go.Scatter(x=recent.index, y=recent[col], mode='lines', showlegend=False,
                       line=dict(color='#4c72b0')),
            row=row, col=pos,
        )
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10),
                       title_text=f'Variables predictoras clave — últimas {weeks} semanas')
    return fig


def main():
    st.title('🚦 Sistema de Alerta Temprana — Estrés Financiero Sistémico')
    st.caption('Predicción del NFCI (Fed de Chicago) · Datos en vivo de la API de FRED')

    model, config, backtest_summary = load_artifacts()
    api_key = get_api_key()

    if not api_key:
        st.warning('Introduce tu API key de FRED en la barra lateral para cargar los datos.')
        st.stop()

    df_weekly, df_features = fetch_and_build_features(
        api_key, config['START_DATE'], config['SERIES'], config['LAGS'], config['ROLLING_WINDOWS'],
    )

    feature_cols = config['FEATURE_COLS']
    latest_valid = df_features.dropna(subset=feature_cols).iloc[[-1]]
    prediction = model.predict(latest_valid[feature_cols])[0]

    threshold_series = df_weekly['nfci'].expanding(
        min_periods=config['MIN_PERIODS_THRESHOLD']
    ).quantile(config['STRESS_PERCENTILE'])
    current_threshold = threshold_series.iloc[-1]
    percentile_rank = (df_weekly['nfci'] < prediction).mean() * 100

    current_nfci = df_weekly['nfci'].iloc[-1]
    current_date = df_weekly.index[-1].date()
    horizon = config['H']

    if prediction > current_threshold:
        status_label, status_color = '🔴 ALERTA — estrés elevado esperado', '#d62728'
    elif prediction > current_threshold * 0.8:
        status_label, status_color = '🟡 VIGILANCIA — condiciones endureciéndose', '#ff9f1c'
    else:
        status_label, status_color = '🟢 NORMAL', '#2ca02c'

    col1, col2, col3 = st.columns(3)
    col1.metric('NFCI actual', f'{current_nfci:.3f}', help=f'Última semana disponible: {current_date}')
    col2.metric(f'NFCI predicho a {horizon} semanas', f'{prediction:.3f}', delta=f'{prediction - current_nfci:+.3f}')
    with col3:
        st.markdown(
            f"<div style='padding-top:8px'><span style='font-size:1.25em;color:{status_color}'>"
            f"<b>{status_label}</b></span></div>",
            unsafe_allow_html=True,
        )

    st.progress(min(max(percentile_rank / 100, 0.0), 1.0),
                text=f'Percentil histórico de la señal predicha: {percentile_rank:.0f}/100 '
                     f'(desde {config["START_DATE"]})')

    st.divider()

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_weekly.index, y=df_weekly['nfci'], name='NFCI real', line=dict(color='black')))
        fig.add_trace(go.Scatter(
            x=threshold_series.index, y=threshold_series.values,
            name=f"Umbral P{int(config['STRESS_PERCENTILE'] * 100)}", line=dict(color='red', dash='dot'),
        ))
        fig.add_hline(y=0, line=dict(color='grey', width=1))
        fig.update_layout(title='Evolución del NFCI y umbral de estrés vigente', height=420,
                           legend=dict(orientation='h', y=1.05))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.plotly_chart(render_driver_chart(model, latest_valid, feature_cols), use_container_width=True)

    st.plotly_chart(render_trend_panel(df_weekly), use_container_width=True)

    st.subheader('Últimos valores de las variables predictoras')
    display_cols = [c for c in df_weekly.columns if c != 'ted_spread']
    st.dataframe(df_weekly[display_cols].tail(8).sort_index(ascending=False), use_container_width=True)

    with st.expander('📋 Metodología y resultados del backtesting histórico'):
        st.markdown(
            'Modelo candidato: **LightGBM** (gradient boosting), comparado contra Ridge y una LSTM, '
            'validado mediante walk-forward (`TimeSeriesSplit`) y evaluado en un holdout final. '
            'El horizonte de predicción es de '
            f'**{horizon} semanas**, y el umbral de estrés es el percentil '
            f'**{int(config["STRESS_PERCENTILE"] * 100)}** de una ventana expansiva del NFCI '
            '(usa solo información pasada en cada momento, igual que aquí).'
        )
        if backtest_summary is not None:
            st.markdown(
                'Resultados del backtesting sobre los tres episodios históricos (reentrenando cada '
                'modelo únicamente con datos anteriores a cada episodio):'
            )
            st.dataframe(backtest_summary, use_container_width=True, hide_index=True)
        st.markdown(
            '**Limitación conocida**: LightGBM, al basarse en árboles, no puede extrapolar más allá '
            'de los valores vistos en entrenamiento — en el backtesting, no detectó la crisis '
            'financiera de 2008 precisamente por ser el primer episodio de esa magnitud en su '
            'historial de entrenamiento. Ver el informe completo para el análisis detallado.'
        )

    st.caption(
        'Ver el informe completo (Word) para la metodología detallada, el backtesting íntegro y '
        'las limitaciones del sistema.'
    )


if __name__ == '__main__':
    main()
