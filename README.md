# Synovia PV Literature Screening

This project converts `PV_Synovia_Pipeline_Colab_Updated_01.ipynb` into a deployable Streamlit Python application. It searches the configured Synovia therapeutic categories in BanglaJOL and PubMed, screens articles for safety relevance, extracts rule-based case fields, prioritizes the review queue, and calculates exploratory PRR/ROR statistics.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first screening run downloads the lightweight Hugging Face model `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`. The model is loaded once with `st.cache_resource`. BanglaJOL and PubMed results are cached with `st.cache_data` for 15 minutes and at most eight query combinations are retained, reducing repeated downloads and CPU use during Streamlit reruns.

## Deploy on Streamlit Community Cloud

Create a GitHub repository containing `app.py`, `requirements.txt`, and this README. In Streamlit Community Cloud, select the repository and set the main file to `app.py`.

## Important limitations

The BanglaJOL and PubMed harvest is cached for 15 minutes, so use the cache-clear option in Streamlit’s app menu or wait for the TTL when fresh results are required. PRR/ROR output is exploratory and is not a substitute for validated pharmacovigilance coding or regulatory review. No audit database or API key is required.
