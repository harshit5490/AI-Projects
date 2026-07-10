What is IQR?

| Plot             | Function            | Data Type                                  | Purpose                                                            | Primary Use                                    | Real-World Example                                   |
| ---------------- | ------------------- | ------------------------------------------ | ------------------------------------------------------------------ | ---------------------------------------------- | ---------------------------------------------------- |
| **Scatter Plot** | `sns.scatterplot()` | Numerical vs Numerical                     | Shows relationship between two numerical variables                 | Detect correlation, trends, clusters, outliers | Age vs Salary, Height vs Weight                      |
| **Line Plot**    | `sns.lineplot()`    | Numerical over Ordered Data                | Shows trends over time or sequence                                 | Time-series analysis                           | Stock prices, Monthly sales, Temperature over days   |
| **Bar Plot**     | `sns.barplot()`     | Category + Numerical                       | Compares the **average** of a numerical variable across categories | Compare mean values                            | Average salary by department, Average marks by class |
| **Histogram**    | `sns.histplot()`    | Numerical                                  | Shows the distribution of a numerical variable                     | Understand spread, skewness, frequency         | Distribution of ages, Salaries, Exam scores          |
| **Box Plot**     | `sns.boxplot()`     | Numerical (optionally grouped by category) | Summarizes distribution and detects outliers                       | Outlier detection, Compare distributions       | Salary distribution across departments               |
| **Count Plot**   | `sns.countplot()`   | Categorical                                | Counts occurrences of each category                                | Frequency analysis                             | Number of males/females, Customers by city           |
| **Pair Plot**    | `sns.pairplot()`    | Multiple Numerical                         | Shows pairwise relationships between numerical features            | Exploratory Data Analysis (EDA)                | Explore relationships in the Iris dataset            |
| **Heatmap**      | `sns.heatmap()`     | Matrix (often correlation matrix)          | Visualizes values using colors                                     | Correlation analysis, Feature selection        | Correlation between house price features             |


| If You Want To...                                     | Use This Plot   |
| ----------------------------------------------------- | --------------- |
| Compare two numerical variables                       | `scatterplot()` |
| Show a trend over time                                | `lineplot()`    |
| Compare average values across categories              | `barplot()`     |
| See how numerical data is distributed                 | `histplot()`    |
| Detect outliers                                       | `boxplot()`     |
| Count categories                                      | `countplot()`   |
| Explore relationships between many numerical features | `pairplot()`    |
| Visualize correlations                                | `heatmap()`     |


| Question                                         | Plot            |
| ------------------------------------------------ | --------------- |
| Is there a relationship between X and Y?         | `scatterplot()` |
| Is sales increasing over time?                   | `lineplot()`    |
| Which department has the highest average salary? | `barplot()`     |
| Are salaries normally distributed?               | `histplot()`    |
| Are there any outliers?                          | `boxplot()`     |
| How many customers belong to each category?      | `countplot()`   |
| Which features are related to each other?        | `pairplot()`    |
| Which features are highly correlated?            | `heatmap()`     |


| Plot            | Most Important Parameters                            |
| --------------- | ---------------------------------------------------- |
| `scatterplot()` | `x`, `y`, `hue`, `style`, `size`                     |
| `lineplot()`    | `x`, `y`, `hue`, `marker`                            |
| `barplot()`     | `x`, `y`, `hue`, `palette`, `errorbar`               |
| `histplot()`    | `x`, `bins`, `kde`, `hue`, `multiple`, `stat`        |
| `boxplot()`     | `x`, `y`, `hue`, `palette`                           |
| `countplot()`   | `x`, `y`, `hue`, `order`, `palette`                  |
| `pairplot()`    | `hue`, `diag_kind`, `corner`, `vars`                 |
| `heatmap()`     | `annot`, `fmt`, `cmap`, `linewidths`, `vmin`, `vmax` |


| Matplotlib                      | Seaborn             | Which One to Prefer?            |
| ------------------------------- | ------------------- | ------------------------------- |
| `plt.scatter()`                 | `sns.scatterplot()` | Seaborn                         |
| `plt.plot()`                    | `sns.lineplot()`    | Seaborn                         |
| `plt.bar()`                     | `sns.barplot()`     | Seaborn (for statistical plots) |
| `plt.hist()`                    | `sns.histplot()`    | Seaborn                         |
| *(No direct equivalent)*        | `sns.boxplot()`     | Seaborn                         |
| *(No direct equivalent)*        | `sns.countplot()`   | Seaborn                         |
| *(No direct equivalent)*        | `sns.pairplot()`    | Seaborn                         |
| `plt.imshow()` (manual heatmap) | `sns.heatmap()`     | Seaborn                         |


| Plot            | Memory Trick                            |
| --------------- | --------------------------------------- |
| 📍 Scatter Plot | **Relationship between two numbers**    |
| 📈 Line Plot    | **Trend over time**                     |
| 📊 Bar Plot     | **Average by category**                 |
| 📉 Histogram    | **Distribution of numbers**             |
| 📦 Box Plot     | **Outliers & spread**                   |
| 🔢 Count Plot   | **Count of categories**                 |
| 🔍 Pair Plot    | **All numerical relationships at once** |
| 🌡️ Heatmap     | **Correlation using colors**            |
