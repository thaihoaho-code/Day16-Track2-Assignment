# Báo Cáo Thực Hành LAB 16: Cloud AI Environment Setup

1. Tôi dùng GCP, us-central1-a, e2-medium (2 vCPU, 4GB RAM), source commit `55539f6`.
2. Dataset Credit Card Fraud có 284,807 dòng, chia train/validation/test 60/20/20, seed 16.
3. Load dữ liệu mất 2.79 giây; training mất 4.72 giây; best iteration của LightGBM là 68.
4. Trên tập test, mô hình đạt AUC: 0.9768, Accuracy: 0.9995, F1: 0.8541, Precision: 0.9080, Recall: 0.8061.
5. Tốc độ suy luận (inference): Latency 1 dòng 1.23 ms; throughput khi chạy batch 1.000 dòng là ~296,348 dòng/giây; cách đo bằng trung vị (`median; warm-up excluded; predict_proba trên pandas input`).
6. CPU/RAM/Network tôi quan sát qua lệnh `top` lúc training đạt đỉnh là `[ví dụ: CPU ~200% (cả 2 core), RAM ~1.5GB]`; xem chi tiết tại ảnh đính kèm `submission\infra submission\screenshots\tai_nguyen_khi_chay.png`.
7. Billing trên GCP tại thời điểm kiểm tra 4h55 2/10 ghi nhận chi phí `chưa cập nhật`; ước tính phí duy trì IP tĩnh và NAT là `[~0.04$/giờ]`.
8. Tôi đã tải toàn bộ kết quả về máy và đã chạy lệnh `terraform destroy` xóa tài nguyên lúc `5h43`; bằng chứng dọn dẹp tại ảnh `submission\infra submission\screenshots\don_dep.png`.
