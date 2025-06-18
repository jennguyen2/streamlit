# Streamlit: Work in Progress

This repository contains several Streamlit applications:

- **Food App**: Helps you browse recipes, create shopping lists, and manage a recipe backlog.
- **Oil App**: Projects Oil prices and notifies when best to buy
- **Profiler App**: 

## Project Structure

```
home.py
food/
    Home.py
    pages/
        1_📒_Get_Recipe.py
        2_🛒_Shopping_List.py
        3_😋_New_Recipe.py
        4_Backlog.py
oil/
    oil_app.py
profiler/
    profiler.py
    pages/
        oneways.py
        oneways_static.py
requirements.txt
```

## Getting Started

### Running the Apps

#### Food App

```sh
streamlit run food/Home.py
```

#### Oil App

```sh
streamlit run oil/oil_app.py
```

#### Profiler App

```sh
streamlit run profiler/profiler.py
```

## Apps

### Food App

- Browse recipes from a dataset.
- View ingredients and instructions.
- Create and manage shopping lists.
- Maintain a backlog of features and ideas.

### Oil App

- Uses Prophet to project oil prices.
- Backlog: identifies a good time to buy and sends an email to notify.

### Profiler App

- Upload CSV or Parquet files, or use a default dataset.
- Filter data by columns and values.
