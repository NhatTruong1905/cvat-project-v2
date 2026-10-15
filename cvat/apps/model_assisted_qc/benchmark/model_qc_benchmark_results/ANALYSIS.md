# Phân tích benchmark Model-assisted QC

## Kết luận

Trên 100 ảnh BDD100K và 1.841 ground-truth boxes, YOLO-BDD100K là lựa chọn tốt nhất
cho referee hiện tại. Model này đạt độ chính xác cao nhất, nhanh nhất, dùng ít VRAM nhất và có
confidence cho từng box để tính Probability Differential (PD).

| Model | mAP50-95 | AP50 | AP75 | AR@100 | Mean latency | FPS | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO-BDD100K | 0.3807 | 0.6559 | 0.3674 | 0.4921 | 39.11 ms | 25.57 | 106.5 MB |
| Grounding DINO tiny | 0.2149 | 0.3722 | 0.2099 | 0.3135 | 625.60 ms | 1.60 | 1,991.9 MB |
| Florence-2 base-ft | 0.1382 | 0.2434 | 0.1204 | 0.1972 | 1,679.85 ms | 0.59 | 643.3 MB |

## Diễn giải

- YOLO có mAP50-95 cao hơn Grounding DINO khoảng 77% và nhanh hơn khoảng 16 lần.
- YOLO có mAP50-95 cao hơn Florence-2 khoảng 176% và nhanh hơn khoảng 43 lần.
- YOLO đặc biệt vượt trội trên vật thể nhỏ: AP small 0.2093, so với 0.0523 của Grounding DINO
  và 0.0182 của Florence-2. Đây là tiêu chí quan trọng với traffic light/sign và đối tượng ở xa.
- Cả ba model có prediction trên gần như toàn bộ tập ảnh: YOLO và Grounding DINO 100/100 ảnh;
  Florence-2 99/100 ảnh.
- Florence-2 `<OD>` không trả confidence theo box. Tất cả 964 detections được gán score 1.0,
  nên không cung cấp tín hiệu confidence phù hợp cho PD và AP ranking của model này chỉ mang tính
  tham khảo.

## Ý nghĩa metrics

- `mAP_50_95`: chỉ số chất lượng chính, trung bình AP qua IoU 0.50 đến 0.95; càng cao càng tốt.
- `AP50`: phát hiện đúng ở điều kiện overlap tương đối dễ (IoU >= 0.50).
- `AP75`: yêu cầu bounding box khớp chặt hơn (IoU >= 0.75), phản ánh localization.
- `AP_small`, `AP_medium`, `AP_large`: AP theo kích thước vật thể.
- `AR_1`, `AR_10`, `AR_100`: average recall khi cho phép tối đa 1, 10 hoặc 100 detection/ảnh.
- `mean_ms`, `median_ms`, `p95_ms`: latency trung bình, trung vị và ngưỡng 95% mỗi ảnh.
- `fps`: số ảnh xử lý mỗi giây; càng cao càng tốt.
- `peak_gpu_memory_mb`: peak VRAM được PyTorch ghi nhận trong inference; càng thấp càng dễ triển khai.
- `predicted_boxes`: số box model xuất ra. Số lượng lớn không đồng nghĩa chính xác cao; COCO AP
  sử dụng confidence để xếp hạng true/false positives.

## Khuyến nghị

Giữ YOLO-BDD100K làm referee production. Grounding DINO chỉ nên được cân nhắc nếu cần thêm lớp
open-vocabulary ngoài taxonomy BDD100K, sau đó cần tuning prompt/threshold trên validation split
riêng. Không dùng Florence-2 `<OD>` trực tiếp cho PD nếu chưa có phương pháp tạo confidence được
calibrate.

Đây là phép so sánh cho bài toán thay thế referee BDD100K: YOLO đã được huấn luyện chuyên biệt
cho taxonomy này, còn Grounding DINO và Florence-2 đang chạy zero-shot. Không nên diễn giải kết quả
thành đánh giá năng lực tổng quát của các foundation model.
