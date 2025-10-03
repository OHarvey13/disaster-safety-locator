# 🌪️ Natural Disaster Safety Location Predictor

## 📖 Overview
When a hurricane or other natural disaster strikes, people often struggle to know where to go for safety. This project uses **deep learning** and **geospatial datasets** to identify the safest places in a region during an emergency. By analyzing historical disaster patterns, shelters, hospitals, flood zones, and infrastructure, the model will recommend **optimal safe zones** for evacuation.

This project is being developed as part of a college semester course on applied machine learning with a focus on **deep learning and real-world impact**.

---

## 🎯 Objectives
- Collect and clean disaster-related datasets (shelters, hospitals, evacuation zones, storm surge maps).
- Build a **deep learning model** to predict safe vs unsafe zones.
- Create a **safety score** for different locations.
- Visualize results with interactive maps and plots.
- Provide a reproducible pipeline for disaster risk modeling.

---

## 📂 Project Structure

```
disaster-safety-predictor/
│── data/                   # Disaster data, maps, shelters, hospitals
│── notebooks/              # Jupyter notebooks for exploration and prototyping
│── src/                    # Source code (model, preprocessing, utils)
│── models/                 # Saved trained models
│── results/                # Metrics, maps, visualizations
│── README.md               # Project documentation
│── requirements.txt        # Python dependencies
│── .gitignore              # Ignored files

```
---


## 📊 Datasets Used

This project integrates multiple public datasets to build a **Hurricane Safety Locator**, focused on Wilmington, NC and surrounding coastal communities.

- **[HURDAT2 Hurricane Database (NOAA)](https://www.aoml.noaa.gov/hrd/hurdat/hurdat2.html)**  
  *Atlantic hurricane tracks and intensity data (1851–present). Provides storm paths, wind speeds, and pressure for modeling storm hazards.*

- **[FEMA National Shelter System (NSS)](https://gis.fema.gov/arcgis/rest/services/NSS/FEMA_NSS/MapServer)**  
  *Shelter locations and capacities. Used to identify safe locations for evacuation and measure shelter accessibility.*

- **[FEMA National Risk Index (NRI)](https://hazards.fema.gov/nri/data-resources)**  
  *Pre-computed county-level risk and resilience scores for natural hazards. Provides vulnerability and community resilience metrics.*

- **(Optional) [FEMA Open Shelters API](https://gis.fema.gov/arcgis/rest/services/NSS/OpenShelters/FeatureServer)**  
  *Real-time feed of currently open shelters. Useful for live demos, but not essential for model training.*
  
---

## 🔄 Data Pipeline Overview

1. **Download HURDAT2 (NOAA)** → Parse storm tracks & intensity.  
2. **Download FEMA NRI CSV** → Extract county-level vulnerability scores.  
3. **Query FEMA NSS API** → Get shelter locations & capacity.  
4. **Merge datasets** on county/city (e.g., Wilmington / New Hanover County).  
5. **Build models** → Predict storm impacts & locate safest evacuation options.  


---

## 🧠 Deep Learning Approach
- **Input Features:** Geospatial layers (risk zones, shelters, hospitals, population density).  
- **Models:**  
  - CNNs or Graph Neural Networks (GNNs) for spatial data.  
  - Classification (safe vs unsafe) or regression (safety score).  
- **Evaluation Metrics:** Accuracy, Precision/Recall, ROC AUC for classification; RMSE for regression.  

---

## 🚀 Setup & Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/disaster-safety-predictor.git
cd disaster-safety-predictor
pip install -r requirements.txt

