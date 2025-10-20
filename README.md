# netflix-data-analysis
This project conducts an introductory data analysis on the "Netflix Titles" dataset. It explores and uncovers insights from Netflix’s content library, identifying trends in movies and TV shows, popular actors, and countries of origin, with visualizations created using Python libraries like Pandas, Matplotlib, and Seaborn.

## Dataset
The dataset used for this project can be found here: [Netflix Shows on Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)

Columns include:  
`show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`

## Technologies Used
- Microsoft Excel  
- Python 3.11.9  
- Pandas  
- Matplotlib  
- Seaborn  

## Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/droche01/netflix-data-analysis.git

2. Navigate to the project directory:
   ```bash
   cd netflix-data-analysis

3. Install dependencies:  
   ```bash
   pip install pandas matplotlib seaborn

4. Run the script:  
   ```bash
   python netflix_analysis.py

## Analysis Highlights

- Visualizations of Netflix content trends over time.  
- Analysis of popular genres and countries producing content.  
- Identification of top actors in movies and TV shows.  
- Runtime analysis for both movies and TV shows.  
- Outlier detection and preparation of a clean dataset.  
- Feature engineering, such as converting durations to minutes and extracting year added, to enable deeper analysis.

## Key Insights and Findings

### Content Distribution
Movies make up the majority of Netflix content, but the number of TV shows has seen a steady increase over time, possibly driven by the popularity of serialized storytelling and shows like *Stranger Things*.

### Popular Content Varies By Region
Dramas are the most popular form of content in the U.S., followed by comedies. Comedies perform well in both the U.S. and India, suggesting that this genre continues to attract strong global interest and could be leveraged to draw more viewers to the platform. In Japan, anime is especially popular, aligning with its cultural significance and dominance in the country's entertainment industry.

### Median Movie Length
The median film length is 98 minutes. Although longer films (around 120 minutes) still attract viewers, the data suggests that the optimal Netflix movie length falls between 90 and 100 minutes -- a duration that balances engagement and accessibility.

### Correlations
There appear to be weak correlations between a title's duration, release year, and the year it was added to Netflix. This indicates that durations have remained relatively stable over time, and that Netflix adds both recently released and older content rather than focusing solely on new releases.
