"""Dashboard de monitorización semanal del Sistema de Alerta Temprana (EWS) de Estrés
Financiero Sistémico. El modelo (LightGBM) y la configuración se entrenan y exportan desde
notebooks/01_datos_eda_feature_engineering.ipynb (Sección 9) — este script solo hace inferencia
sobre datos frescos de FRED, sin reentrenar nada.
"""

import json
import os
from datetime import date

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fredapi import Fred

st.set_page_config(page_title='EWS Estrés Financiero', page_icon='🚦', layout='wide')

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(ARTIFACTS_DIR, 'lgbm_model.pkl'))
    with open(os.path.join(ARTIFACTS_DIR, 'config.json')) as f:
        config = json.load(f)
    return model, config


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


def main():
    st.title('🚦 Sistema de Alerta Temprana — Estrés Financiero Sistémico')
    st.caption('Predicción del NFCI (Fed de Chicago) · Datos en vivo de la API de FRED')

    model, config = load_artifacts()
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

    st.divider()

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

    st.subheader('Últimos valores de las variables predictoras')
    display_cols = [c for c in df_weekly.columns if c != 'ted_spread']
    st.dataframe(df_weekly[display_cols].tail(8).sort_index(ascending=False), use_container_width=True)

    st.caption(
        'Modelo: LightGBM (gradient boosting), validado mediante walk-forward y backtesting sobre '
        'GFC 2008, COVID 2020 y SVB 2023 — ver informe completo para metodología y limitaciones '
        '(en particular, capacidad reducida ante eventos de magnitud sin precedentes en el histórico '
        'de entrenamiento).'
    )


if __name__ == '__main__':
    main()
