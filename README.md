# house_sales_time_series

## Preprocessing steps
- Change the columns names
- Fill the missing values in bedrooms with the median
- Add new column to indentify outliers in the `price`
- Add time based feautres, `day`, `year`, `month`
- encoding of the data
    - `bedroom` (lebel encoding)
    - `post_code`, `property_type` (one hot encoding)
- Transform the data with scaling, then logarithmic transformation