# E-commerce customer segmentation

K-Means segmentation of an international e-commerce customer base, completed for Cambridge PACE Course 101 (week 6).

The brief was to turn a large order-level table into customer groups that marketing could actually use. The source is a transnational extract covering 47 countries and orders from 1 January 2012 to 30 December 2016 (SAS, 2024). After collapsing orders to one row per customer, the clustering matrix has 68,300 customers and five features.

## Data

The course published the extract in a public dataset repository. The notebook downloads it from:

https://github.com/FourthRevGitHub/CAM_DS_Public-Data-Sets/raw/main/Course%201/Week%206/CUSTOMERS_CLEAN.zip

The zip is not stored in this folder. Columns include order and customer fields such as quantity, dates, revenue, unit cost, profit, country, loyalty flags and customer id. Some commercial fields are incomplete (for example, profit, total revenue, unit cost and quantity have tens of thousands of missing values in the raw table). Location fields also have gaps (`state_province` 117,192 missing; `postal_code` 3,716; `city` 135). The five modelling features were built after aggregation, with median imputation on the numeric clustering matrix.

## Approach

1. Clean types, scan missing values and duplicates, then aggregate to customer level.
2. Engineer five features chosen in the course brief: purchase frequency, recency (days since last order), customer lifetime value, average unit cost, and age derived from date of birth.
3. Standardise the matrix.
4. Choose *k* with three checks together: the elbow curve, silhouette scores, and a hierarchical dendrogram. The elbow heuristic pointed to 6 and the silhouette peak pointed to 5; the dendrogram also favoured a small number of groups. The final model uses **k = 5**.
5. Fit K-Means (`random_state=42`) and profile each segment with means and boxplots.
6. Project the same labels with PCA and t-SNE as a visual check, not as the clustering itself.

## Results

Segment sizes and mean profiles, taken from the executed notebook:

| Segment | Customers | Frequency | Recency (days) | CLV | Avg unit cost | Age |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10,660 | 4.02 | 1,115 | 610 | 63.65 | 48.0 |
| 1 | 11,245 | 25.80 | 122 | 4,899 | 82.70 | 39.4 |
| 2 | 22,510 | 9.87 | 231 | 1,652 | 74.13 | 28.5 |
| 3 | 3,804 | 4.33 | 574 | 1,583 | 180.43 | 45.1 |
| 4 | 20,081 | 8.78 | 227 | 1,414 | 70.96 | 60.3 |

A plain reading of those means:

- **Segment 1** is the high-value, high-frequency group with recent activity.
- **Segment 0** looks inactive: very long recency, low frequency and the lowest CLV.
- **Segment 3** is small and buys expensive items (average unit cost 180) but not often, and has not purchased as recently as the core groups.
- **Segments 2 and 4** are the two large “regular” groups, split mainly by age (about 29 vs 60) with similar recency and mid-range CLV.

The numeric silhouette scores were computed and plotted but not printed as a table in the notebook, so they are not quoted here.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/customer_segmentation.ipynb
```

The first data cell downloads `CUSTOMERS_CLEAN.zip` from the public course URL. You need a network connection for that step.

## Tech stack

Python, pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn.

## Files

- `notebooks/customer_segmentation.ipynb` — full analysis with executed outputs

## Reference

SAS, 2024. CUSTOMERS_CLEAN [Data set]. SAS. Last revised on 15 December 2021.
