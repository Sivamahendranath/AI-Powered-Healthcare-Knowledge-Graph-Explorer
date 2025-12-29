# AI-Powered-Healthcare-Knowledge-Graph-Explorer
AI-powered Streamlit app for extracting, storing, and visualizing healthcare entities and their relationships using Gemini and SQLite.




An **AI-powered Streamlit application** that extracts healthcare-related entities (hospitals, clinics, doctors, services, locations, and service times) from text and URLs using the **Google Gemini API**, stores them in **SQLite**, and visualizes relationships through interactive graphs.

---

## 📌 Overview

This project focuses on **healthcare domain entity extraction**, with **hospitals and clinics treated as high-priority entities**. The application parses AI-generated responses, identifies structured attributes and relationships, and builds a persistent, queryable knowledge graph.

All functionality in this repository directly reflects the provided source file.

---

## ✨ Key Features

* Healthcare-focused entity extraction using Gemini
* Priority handling for hospitals and clinics
* Extraction of attributes such as:

  * Location and address
  * Service hours
  * Pharmacy and ambulance availability
  * Specializations and services
  * Doctor experience, availability, and fees
* Relationship detection and normalization between entities
* SQLite database with indexed tables for:

  * Entities
  * Relationships
  * Detailed attributes
* Dynamic entity and edge weight calculation
* Interactive Streamlit user interface
* Graph-based visualization using NetworkX and PyVis
* CSV export of extracted entities

---

## 🧠 How It Works

1. **Input**

   * User provides text content or a URL
2. **AI Processing**

   * Text is sent to the Gemini model with a healthcare-specific extraction prompt
3. **Parsing**

   * AI responses are parsed into entities, attributes, and relationships
4. **Storage**

   * Data is stored in a structured SQLite database
5. **Analysis**

   * Entities and relationships are weighted based on connectivity and detail richness
6. **Visualization**

   * Results are displayed in tables and interactive graphs in Streamlit

---

## 🗂 Project Structure

```text
.
├── project.py                # Main Streamlit application
├── entities.db           # SQLite database (generated at runtime)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore
├── README.md
└── LICENSE
```

> The core logic includes entity parsing, relationship extraction, database management, weighting algorithms, and graph expansion logic, all derived from the original source file.

---

## 🛠 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Sivamahendranath/AI-Powered-Healthcare-Knowledge-Graph-Explorer
cd Sivamahendranath/AI-Powered-Healthcare-Knowledge-Graph-Explorer
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API=your_gemini_api_key
```

---

## ▶️ Running the Application

```bash
streamlit run Project.py
```

The application will launch in your browser.

---

## 🗄 Database

* SQLite database file: `entities.db`
* Automatically created and updated at runtime
* Stores:

  * Entities
  * Relationships
  * Detailed attributes
  * Calculated weights

---

## 📤 Data Export

* Extracted entity data can be downloaded as a **CSV file** directly from the UI.

---

## ⚠️ Notes

* This repository contains **no additional features** beyond what is implemented in the provided source file.
* Internet access is required for:

  * Gemini API usage
  * URL content extraction
* API rate limiting and retry logic are built into the application.

---

## 📜 License

This project is licensed under the **MIT License**.

---

