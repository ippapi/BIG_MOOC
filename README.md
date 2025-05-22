

# BIG_MOOC
## **Project Overview**
This project is built to handle **real-time data ingestion**, **DL model training using BIG DATA**, and **recommendation** serving with a robust, scalable architecture. 
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
	 - 📁 database_initiator (cloud Cassandra data uploader)
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
	 - 📁 notebooks (notebook to run model on colab)
	 - 📁 batch_process 
		 - 📁 BERT4Rec (contains files to train BERT4Rec model)
		 - 📁 FM (contains files to train FM model)
		 - 📁 notebooks (contains notebooks to run model)
		 - load_recommendations.py (load recommendations predicted by model to database)
		 - recommendations.csv (contains)
	 - README.md: It's me :))
---
