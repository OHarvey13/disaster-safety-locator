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


## 📊 Potential Datasets
- **FEMA National Risk Index** – Community-level hazard risk & vulnerability.  
- **FEMA National Shelter System (NSS)** – Locations and attributes of shelters.  
- **NOAA / NHC** – Historical hurricane paths and storm surge data.  
- **OpenStreetMap (OSM)** – Hospitals, roads, and shelter infrastructure.  
- **Local / State Emergency Management Data** – Evacuation zones, flood zones.  

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

