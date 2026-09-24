# Trạng thái dự án

Cập nhật: 2026-09-24. **Sáu cổng đã hoàn thành.**

| Cổng | Nội dung | Kết quả |
|---|---|---|
| 0 | Đo nhiễu của phép đo | nhiễu 8,5%, ngưỡng tuyên bố cải thiện 25,5% |
| 1 | Bộ đo độ chính xác + độ trễ | đường cơ sở 64,7% ChartQA / 990 ms |
| 2 | Đòn bẩy giảm token ảnh | giảm cạnh 1536→768: **1,93×**, mất 7,3 điểm (p = 0,002) |
| 3 | Lượng tử hoá | nf4 giảm **55% VRAM**, chậm 8%, có dấu hiệu mất 4 điểm (p = 0,081) |
| 4 | Máy chủ gRPC | cạnh 768 cho **2,02 yêu cầu/giây**, gấp đôi bản gốc |
| 5 | Kiểm tra tỉnh táo + ablation prompt | DocVQA ANLS 73,8 so với 81,6 đã công bố; ngôn ngữ chỉ dẫn không ảnh hưởng |

Kèm theo: 38 test, hai notebook dạy, `speedrun.sh` đã chạy thử trọn vẹn.

Hạ tầng đã kiểm chứng:

| Thành phần | Trạng thái |
|---|---|
| Bản clone mới chạy được máy chủ | ✅ mã gRPC tự sinh lại khi thiếu hoặc lệch phiên bản |
| Môi trường CI (`pixi run test`) | ✅ xanh trên GitHub: 38 test, 33 giây |
| Ảnh Docker | ✅ 3,49 GB, build chịu được mạng chập chờn |
| Docker có GPU | ✅ chạy và đo được; chênh với chạy trực tiếp 3–9%, dưới ngưỡng nhiễu |

## Phần mở rộng có thể làm thêm

Không bắt buộc, xếp theo mức đáng làm:

1. **Cắt token theo điểm attention (kiểu FastV)** — bốn cách chọn hiện tại đều
   không dùng thông tin từ mô hình. Cần can thiệp vào vòng lặp decoder.
2. **Ghép cạnh 768 với nf4 rồi đo lại phần phục vụ** — hiện mới đo riêng từng cái
   ở phần ngoại tuyến.
3. **Thêm một mô hình thứ hai** (ví dụ Qwen2.5-VL-3B) để kiểm tra kết luận "bộ mã
   hoá thị giác chiếm hơn một nửa thời gian" có đúng ngoài SmolVLM không.
4. **Batching ở máy chủ** — hiện mỗi lần chỉ xử lý một yêu cầu; gộp lô có thể tăng
   thông lượng mà không đổi phần cứng.

## Lưu ý khi chạy lại

- Không chạy hai phép đo cùng lúc trên một GPU: độ trễ vọt từ 904 ms lên 5.247 ms.
- Luôn truyền `--model`; mặc định đã đổi sang bản 2.2B nhưng vẫn nên ghi rõ.
- `FAST=1 ./speedrun.sh` ghi đè biểu đồ bằng dữ liệu 12 mẫu. Sau khi chạy nhanh,
  vẽ lại bằng: `pixi run python -m bench.plot --results results/gate2_sweep.json`
- Không dùng `pkill -f` với chuỗi trùng nội dung lệnh đang gõ, nó sẽ tự giết shell.
