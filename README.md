<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

# BigMOOC

## Giới Thiệu Tổng Quan

BigMOOC là dự án được phát triển qua hai môn học Mạng Xã Hội - (IS353) được cô Nguyễn Thị Anh Thư và Công Nghệ Dữ Liệu Lớn - IE212 và thầy Hà Minh Tân hướng dẫn của nhóm sinh viên của trường Đại học Công nghệ Thông tin - ĐHQG Thành phố Hồ Chí Minh. Đồ án tập trung xây dựng hệ khuyến nghị khóa học trên nền tảng trực tuyến (MOOC) trong đó xây dựng và thiết kế ứng dụng xử lý các tương tác thời gian thực với các công nghệ dữ liệu lớn.

## Thành Viên Nhóm

Các thành viên trong nhóm bao gồm:

| STT | Tên                  | Mã Số Sinh Viên | Vai Trò     |
| --- | -------------------- | --------------- | ----------- |
| 1   | Vũ Thanh Phong       | 22521095        | Trưởng nhóm |
| 2   | Phạm Trung Nguyên    | 22520983        | Thành Viên  |
| 3   | Võ Hoàng Thảo Phương | 22521171        | Thành Viên  |
| 4   | Ngô Phương Quyên     | 22521221        | Thành Viên  |

## Công nghệ sử dụng

 - Frontend: EJS, CSS.
 - Backend: FastAPI (Python), Express (Node.js).
 - Database: Apache Cassandra.
 - Recommendation System: PyTorch
 - Big Data Processing: Spark
 - Streaming: Apache Kafka, Ngrok
 - Containerization: Docker
 - Công cụ khác: Colab, Kaggle, Draw.io,...

[Drive](https://drive.google.com/drive/u/2/folders/1naMgCV6hGWTB25WFfiNL0xF-kCNGqjwx)

## Cài Đặt

### Hướng Dẫn Cài Đặt

**Bước 1:** Clone repo về máy tính của bạn bằng cách sử dụng git command line hoặc download zip file.

```bash
git clone https://github.com/ippapi/BIG_MOOC.git
```

**Bước 2:** Chạy producer docker images
```bash
docker-compose up --build producer
```

**Bước 3:** Khởi tạo fastapi app + ngrok trên colab
```bash
docker-compose up --build consumer
```

**Bước 4:** Cập nhật ngrok link trong thư mục producer main.py và khởi chạy producer docker images

**Bước 5:** Khởi tạo node.js app.

**Bước 6:** Chạy spark để nhận và xử lý dữ liệu streaming.

**Bước 7:** Mở cổng kết nối ngrok nhận dữ liệu trả về.
```Bash
ngrok http --url=careful-formally-locust.ngrok-free.app 8000
```

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
