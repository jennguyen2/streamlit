# Streamlit Food & Oil Apps

This repository contains two Streamlit applications:

- **Food App**: Helps you browse recipes, create shopping lists, and manage a recipe backlog.
- **Oil App**: Allows you to upload CSV datasets and view summary statistics using pandas.

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
profiling/
    home.py
requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.8+
- See [requirements.txt](requirements.txt) for dependencies.

### Installation

1. Clone the repository:
    ```sh
    git clone <repo-url>
    cd streamlit
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Apps

#### Food App

```sh
streamlit run food/Home.py
```

#### Oil App

```sh
streamlit run oil/oil_app.py
```

#### Pandas Profiling App

```sh
streamlit run home.py
```

## Features

### Food App

- Browse recipes from a dataset.
- View ingredients and instructions.
- Create and manage shopping lists.
- Maintain a backlog of features and ideas.

### Oil App

- Uses Prophet to project oil prices
- Backlog: identifies a good time to buy and sends an email to notify.

## Notes

- The Food App loads a dataset from HuggingFace (`hf://datasets/Hieu-Pham/kaggle_food_recipes/...`). Ensure you have access and the required packages (`fsspec`, `s3fs`, `huggingface_hub`) installed if needed.
- For additional dependencies, see [oil/requirements.txt](oil/requirements.txt).

---

Feel free to contribute or open issues!