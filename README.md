# BIG_MOOC
## **Project Overview**
This project is built to handle **real-time data ingestion**, **ML model training**, and **recommendation** serving with a robust, scalable architecture. Here’s a breakdown of the system components:
 -  **Database Node**: The `database_node` folder sets up a **Cassandra database** to store and manage user event data. This is crucial for storing large-scale event data, which is used in training the ML models.
 -  **Kafka**: Kafka handles **real-time event streaming**, ensuring that data from various sources (like user activity or logs) are pushed into the system efficiently and can be consumed by different services like the ML pipeline.
 -  **ML Node**: The `recommend_node` folder contains the logic for training the recommendation model, as well as serving predictions for real-time recommendation requests.
 -  **App Node**: The `app` folder contains APIs that interact with **Kafka**, **Cassandra**, and the **recommendation system**. It listens to Kafka topics, writes data to Cassandra, and exposes the ML predictions as a service.
---
## **Project resource**
- Database Node images: ippapi/database_node
- Notebooks + database's data: https://drive.google.com/drive/u/2/folders/1naMgCV6hGWTB25WFfiNL0xF-kCNGqjwx
## **Project file structure**

 - BIG_MOOC
	 - 📁 app (FastAPI app + Kafka consumers/producers = Backend APIs)
		 - main.py (API entrypoint)
		 - kafka_consumer.py (Listen to Kafka, write to Cassandra)
	 - 📁 database_node (Cassandra setup = Docker + init scripts)
		 - 📁 data (table's data in csv format)
		 - 📁 binded_mount (binded mount data from cassandra db)
			 - user.csv (user table)
			 - course.csv (course table)
			 - user_course.csv (user and course interaction with timestamp)
		 - init.cql (Create init tables for keyspace)
		 - data_loader.py (Load data from data directory to db)
		 - requirements.txt (Python dependencies)
		 - Dockerfile (Setting up data loader app)
	 -  📁 recommend_node
	 - 📁 kafka
	 - 📁 notebooks
	 - 📁 utils
	 - Dockerfile
	 - README.md: It's me :))

  

---

