# Project: World Bank Educational Data Analysis
## International Expansion Strategy for academy EdTech Platform

[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.12+-yellow.svg)](https://www.python.org/)

###  Project Context
This project analyzes World Bank EdStats data to identify international expansion opportunities for academy's EdTech platform. The goal is to determine which countries have the most potential for educational technology services based on various indicators.

###  Business & Technical Objectives
- **Identify Key Indicators**: Determine which educational and economic metrics best predict EdTech market potential.
- **Data Quality Assessment**: Evaluate the completeness and reliability of global educational data.
- **Market Prioritization**: Rank countries based on a weighted score of potential, considering infrastructure, demographics, and investment.

###  Technical Architecture
The project follows a structured data analysis workflow:
1. **Data Loading**: Efficiently loading multiple CSV files from the World Bank dataset.
2. **Data Cleaning**: Handling missing values and filtering relevant indicators.
3. **Exploratory Data Analysis (EDA)**: Statistical analysis of educational trends.
4. **Visualization**: Using Plotly for interactive and web-ready charts.

---

###  Quick Start (Docker)

The environment is containerized for easy reproduction.

#### 1. Prerequisites
- Docker Desktop
- Docker Compose V2

#### 2. Launch the System
```bash
docker-compose up --build
```

#### 3. Access the Services
- **Jupyter Notebook**: [http://localhost:8882](http://localhost:8882) (Open mission2.ipynb)

---

###  Project Structure
```text
 mission2.ipynb       # Main analysis notebook
 src/
    classes/         # Data processing classes
    scripts/         # Analysis scripts
    utils/           # Utility functions
 dataset/             # World Bank CSV files
 docker-compose.yml   # Container orchestration
 Dockerfile           # Python environment
```

###  Key Insights
- **Infrastructure is King**: Internet penetration and electricity access are the most critical prerequisites for EdTech adoption.
- **Demographic Dividends**: Countries with a large youth population and rising secondary education enrollment show the highest long-term growth potential.
- **Data Gaps**: Many emerging markets have significant data gaps, requiring robust imputation or proxy indicators for accurate assessment.

---
*This project demonstrates the ability to perform large-scale data analysis and provide strategic business recommendations based on global educational data.*
