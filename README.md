# TFM — Sistema de Alerta Temprana (EWS) de Estrés Financiero Sistémico

Máster en Ciencia de Datos. Predicción anticipada del NFCI (Fed de Chicago) mediante ML
sobre series macro/mercado de FRED, con backtesting sobre GFC 2008, COVID 2020 y SVB 2023.

## Estructura del proyecto

```
TFM/
├── notebooks/
│   └── 01_datos_eda_feature_engineering.ipynb   # todo el código: datos, modelos, backtesting, SHAP
├── streamlit_app/
│   ├── app.py                                   # dashboard de monitorización semanal
│   ├── requirements.txt
│   └── artifacts/                               # modelo + config exportados desde el notebook (Sección 9)
│       ├── lgbm_model.pkl
│       └── config.json
├── data/
│   ├── raw/                                     # caché de series descargadas de FRED (csv)
│   └── processed/                               # dataset con features listo para modelizar
└── README.md
```

## Cómo trabajar en Google Colab

1. Sube la carpeta `notebooks/` a tu Google Drive (o abre el `.ipynb` directamente en
   [colab.research.google.com](https://colab.research.google.com) → "Subir").
2. Consigue una API key gratuita de FRED: https://fred.stlouisfed.org/docs/api/api_key.html
3. En Colab, panel lateral izquierdo → icono de llave 🔑 ("Secrets") → añade un secreto
   llamado `FRED_API_KEY` con tu clave, y activa "acceso al notebook".
4. Ejecuta las celdas en orden. Los datos crudos y procesados se guardan en `data/` dentro
   del entorno de Colab (efímero); si quieres conservarlos entre sesiones, monta Google Drive
   o descarga los CSV generados.

## Dashboard (Streamlit)

El dashboard **no reentrena nada**: carga el modelo LightGBM y la configuración exportados por
la Sección 9 del notebook (`artifacts/lgbm_model.pkl` y `artifacts/config.json`), descarga datos
frescos de FRED en cada visita, y calcula la señal de la semana.

### Pasos

1. En Colab, tras ejecutar la Sección 9, descarga la carpeta `artifacts/` (panel de archivos
   izquierdo → botón derecho sobre `artifacts` → Descargar) y colócala dentro de
   `streamlit_app/` en este proyecto, tal como muestra el árbol de arriba.
2. Despliega con **Streamlit Community Cloud** (recomendado — no requiere Python instalado
   localmente):
   - Sube esta carpeta `TFM/` a un repositorio de GitHub (puede ser privado).
   - Entra en [share.streamlit.io](https://share.streamlit.io), conecta el repo y selecciona
     `streamlit_app/app.py` como archivo principal.
   - En "Advanced settings → Secrets", añade:
     ```toml
     FRED_API_KEY = "tu_clave_aqui"
     ```
   - Despliega. Obtendrás una URL pública para incluir en el informe y en el vídeo.
3. Alternativa sin GitHub (prueba rápida desde el propio Colab): instalar `streamlit` y
   `pyngrok`/`localtunnel` en una celda del notebook para exponer un túnel temporal — útil solo
   para grabar el vídeo de demo, no como entrega final.

## Estado

- [x] Sección 2 — Datos, EDA y feature engineering
- [x] Sección 3 — Modelización (Ridge, LightGBM walk-forward, LSTM)
- [x] Sección 4 — Backtesting (GFC 2008, COVID 2020, SVB 2023)
- [x] Sección 5 — Interpretabilidad (SHAP)
- [x] Sección 6 — Dashboard Streamlit (código listo; pendiente descargar artefactos y desplegar)
- [ ] Informe Word (20 caras) y vídeo de presentación
