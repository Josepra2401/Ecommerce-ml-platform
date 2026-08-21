# 📦 E-Commerce ML Platform

Final Year Project — Integrated ML analytics platform for e-commerce.

## Modules

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit entry point |
| `pipeline.py` | Full ML pipeline (run all 3 modules at once) |
| `modules/data_loader.py` | File upload, column auto-detection |
| `modules/forecasting.py` | Prophet sales forecasting |
| `modules/segmentation.py` | RFM + KMeans customer segmentation |
| `modules/geo_analysis.py` | City bar chart + Folium map |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run Pipeline Directly

```bash
python pipeline.py your_data.csv
```

## Dataset Columns Expected

`order_id`, `product_id`, `description`, `quantity`, `order_date`, `price`, `user_id`, `city`, `total_price`
