

# BIG_MOOC
## **Project Overview**
This project is built to handle **real-time data ingestion**, **ML model training**, and **recommendation** serving with a robust, scalable architecture. Here’s a breakdown of the system components:
 -  **Database Node**: The `database_node` folder sets up a **Cassandra database** to store and manage user event data. This is crucial for storing large-scale event data, which is used in training the ML models.
 -  **Kafka**: Kafka handles **real-time event streaming**, ensuring that data from various sources (like user activity or logs) are pushed into the system efficiently and can be consumed by different services like the ML pipeline.
 -  **REC Node**: The `recommend_node` folder contains the logic for training the recommendation model, as well as serving predictions for real-time recommendation requests.
 -  **App Node**: The `app` folder contains APIs that interact with **Kafka**, **Cassandra**, and the **recommendation system**. It listens to Kafka topics, writes data to Cassandra, and exposes the ML predictions as a service.
---
## **Project resource**
- **Notebooks + database's data**: https://drive.google.com/drive/u/2/folders/1naMgCV6hGWTB25WFfiNL0xF-kCNGqjwx
---
## **Project file structure**

 - BIG_MOOC
	 - 📁 app (FastAPI app + Kafka consumers/producers)
		 - 	kafka_producer (Listen to client, write data to broker)
		 - kafka_consumer (Listen to broker and tranfer data to colab notebook)
		 - readme.md: info about consumer and producer
	 - 📁 database_node (cloud Cassandra data uploader)
		 - 📁 data (table's data in csv format)
			 - user.csv (user table)
			 - course.csv (course table)
			 - user_course.csv (user and course interaction with timestamp)
			 - course_map.csv (course mapping for training process)
		 - data_loader.py (Load data from data directory to db)
	 -  📁 stream_process
		 - pretrain_model (function, class to train model offline)
		 - ddp_worker.py (worker on streaming batch)
		 - main.py (main program to run distributed model and streaming data)
		 - model.py (contains model to train both online and offline)
	 - 📁 web
		 - public (contains web media)
		 - views (contains ejs file build web ui/ux + interact logic)
		 - app.js (main app)
	 - 📁 notebooks
	  	 - data processing: for preprocessing, cleaning, and mapping input data before modeling or analysis.
		 - STREAMING_BIG_MOOC: notebook to run model on colab
	 - README.md: It's me :))
---
