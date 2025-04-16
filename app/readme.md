# Real-Time data streaming to colab with kafka + fastapi
## **Folder Overview**
 -  **Consumer**: Nhận (subscribe) events của các topic ( các cặp key, value lưu dưới dạng binary). Các event sẽ được nhận qua địa chỉ quảng bá của kafka broker: kafka:9092. Mỗi event được đọc từ topic partition trong broker chính (thực ra có 1 broker à hhh) phân tích cú pháp (parsing) rồi gửi địa chỉ được quảng bá bằng ngrok trên colab notebook.
 -  **Producer**: Ghi dữ liệu vào topic partition trong leader broker (các events cùng key sẽ được đảm vào thuộc cùng 1 partition). Giao tiếp với consumer qua địa chỉ quảng bá kafka:9092.
 - Cái nghèo dạy ta lớn, cái đói nuôi ta khôn.
---

