# Stock Value Analysis Project

This project involves analyzing daily closing stock prices for five companies: Exxon (XOM), Canadian Solar (CSIQ), Intel (INTC), Walmart (WMT), and Palantir (PLTR). 
The analysis leverages Python's Pandas library for data manipulation and Matplotlib for visualization.

## Features
- **Data Loading**: Reads stock price data from a CSV file.
- **Data Preprocessing**: Converts date columns to datetime format and sets the date as the index for time-series analysis.
- **Descriptive Statistics**: Provides a summary of the data including size, dimensions, and data types.
- **Data Slicing**: Extracts specific rows and columns using both `.iloc` and `.loc` methods.
- **Growth Calculation**: Computes daily growth rates of stock prices.
- **Correlation Analysis**: Calculates Pearson correlation coefficients between the stock prices.
- **Visualization**: Plots line graphs to show monthly average growth rates of the stocks.

### Prerequisites and Modules Used

- Python 3.x
- Pandas
- Matplotlib

### Project Structure

- `stockvalueanalysis.ipynb`: Main notebook containing the analysis.
- `Stock_Data.csv`: Sample data file used for analysis.
- `README.md`: Project description and instructions.
