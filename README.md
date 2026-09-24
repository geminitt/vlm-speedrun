<div align="center">

# vlm-speedrun

**Chạy mô hình thị giác–ngôn ngữ trên GPU 6 GB: thời gian thực sự nằm ở đâu, và đòn bẩy nào đáng kéo**

</div>

---

## Kết quả một dòng

> Trên GPU laptop 6 GB, **bộ mã hoá thị giác chiếm 52,7% thời gian**, còn phần mô
> hình ngôn ngữ xử lý hơn một nghìn token ảnh chỉ chiếm 29,3%. Vì vậy **cắt token
> ảnh sau khi mã hoá — kỹ thuật được nhắc tới nhiều nhất — chỉ đạt 1,13–1,26×**,
> trong khi **giảm độ phân giải đầu vào đạt 1,93× và tăng gấp đôi thông lượng dịch
> vụ**, đổi lại 7,3 điểm độ chính xác. Lượng tử hoá nf4 **giảm 55% bộ nhớ**, nhưng
> chậm hơn 8% và có dấu hiệu mất khoảng 4 điểm (p = 0,081) — nó mua bộ nhớ, không
> mua tốc độ.

![Đánh đổi tốc độ và chất lượng](results/tradeoff_light.png)

## Chạy lại toàn bộ bằng một lệnh

```bash
./speedrun.sh            # đầy đủ, khoảng 45–60 phút
FAST=1 ./speedrun.sh     # rút gọn để kiểm tra pipeline, khoảng 6 phút
```

Phần cứng dùng để đo: **NVIDIA RTX 1000 Ada Laptop, 6 GB, compute capability 8.9**,
chạy trong WSL2. Không cần GPU đám mây, không tốn tiền API.
Mô hình: **SmolVLM-2.2B**. Dữ liệu: **ChartQA** (chấm theo *relaxed accuracy*) và
**DocVQA** (chấm theo *ANLS*, dùng cho phép kiểm tra tỉnh táo).

---

## Bảng kết quả

### Thời gian nằm ở đâu

| Thành phần | Thời gian | Tỉ lệ |
|---|---:|---:|
| **Bộ mã hoá thị giác** | 453 ms | **52,7%** |
| Connector | 2,6 ms | 0,3% |
| Prefill của mô hình ngôn ngữ | 252 ms | 29,3% |
| Sinh 32 token | 142 ms | 16,5% |
| *(tiền xử lý ảnh phía CPU)* | *29 ms* | *3%* |

Một ảnh 800×557 bị cắt thành **13 ô**, sinh ra **1.053 token ảnh** — chiếm 84% toàn
bộ chuỗi đầu vào.

### Đòn bẩy giảm token ảnh

Cột p là kiểm định McNemar theo cặp so với bản gốc.

| Cấu hình | Token ảnh | Độ chính xác | Δ | p | Tăng tốc |
|---|---:|---|---:|---:|---:|
| Bản gốc (13 ô, cạnh 1536) | 1.053 | 64,7% | — | — | 1,00× |
| **Giảm cạnh dài còn 768** | 405 | 57,3% | −7,3 | **0,002** | **1,93×** |
| Một ô duy nhất, không cắt | 81 | 41,0% | −28,0 | <0,0001 | 2,08× |
| Cắt token còn 50%, cách đều | 526 | 42,5% | — | — | 1,13× |
| Cắt token còn 25%, cách đều | 263 | 30,0% | — | — | 1,20× |
| Cắt token còn 25%, **ngẫu nhiên** | 263 | 31,7% | — | — | 1,23× |
| Cắt token còn 25%, gộp trung bình | 264 | 25,0% | — | — | 1,23× |

Hai kết luận:

- **Giảm số ô ảnh thắng cắt token ở cả hai mặt.** Cùng cho ra 81 token ảnh, cách
  "một ô duy nhất" nhanh 2,08× và đúng 41,0%, còn cách "cắt token còn 7,7%" chỉ
  nhanh 1,26× và đúng 22,5%. Cắt token sau khi mã hoá vẫn phải trả toàn bộ chi phí
  mã hoá thị giác.
- **Chọn token nào không quan trọng, chỉ số lượng mới quan trọng.** Cách đều cho
  30,0%, chọn ngẫu nhiên cho 31,7% ở cùng số token — không phân biệt được. Nhánh
  đối chứng ngẫu nhiên tồn tại chính để trả lời câu này.

### Lượng tử hoá

| Chế độ | Độ trễ | VRAM đỉnh | Khớp token đầu | Lệch logit tối đa |
|---|---:|---:|---:|---:|
| bf16 (chuẩn) | 907 ms | 4.805 MB | 100% | 0 |
| fp16 | 927 ms | 4.805 MB | 100% | 5,97 |
| int8 | **2.340 ms** | 3.008 MB | 100% | 3,16 |
| **nf4** | 922 ms | **1.905 MB** | 100% | 6,78 |

**Lượng tử hoá ở đây mua bộ nhớ, không mua tốc độ.** int8 của bitsandbytes còn chậm
hơn 2,6× vì chi phí giải lượng tử trong mỗi phép nhân ma trận.

Đo kỹ nf4 trên **300 mẫu**, so theo cặp với bf16 trên cùng tập mẫu:

| | bf16 | nf4 | chênh lệch |
|---|---:|---:|---|
| Độ chính xác (bản gốc) | 64,7% | 60,7% | −4,0 điểm, **p = 0,081** |
| Độ chính xác (cạnh 768) | 57,3% | 54,3% | −3,0 điểm, p = 0,200 |
| Độ trễ | 995 ms | 1.081 ms | **chậm hơn 8%** |
| VRAM đỉnh | 5.213 MB | **2.321 MB** | **−55%** |

Ở mức p = 0,081 thì chưa đủ bằng chứng theo ngưỡng 0,05, nhưng đã sát — nên cách
nói đúng là *"có dấu hiệu mất khoảng 4 điểm"*, không phải *"không ảnh hưởng"*.

### Phục vụ qua gRPC

| Cạnh dài | Đồng thời | Trung vị | p95 | Thông lượng |
|---:|---:|---:|---:|---:|
| 1536 | 1 | 939 ms | 1.201 ms | 1,03 yc/s |
| 1536 | 2 | 1.931 ms | 2.442 ms | 1,02 yc/s |
| 1536 | 4 | 3.848 ms | 4.342 ms | 1,01 yc/s |
| **768** | 1 | **503 ms** | 635 ms | **2,02 yc/s** |
| 768 | 2 | 993 ms | 1.111 ms | 1,99 yc/s |
| 768 | 4 | 1.927 ms | 2.111 ms | 1,99 yc/s |

Thời gian tính tại máy chủ **không đổi theo mức đồng thời**; toàn bộ phần tăng thêm
là chờ hàng đợi. Một GPU phục vụ một yêu cầu tại một thời điểm, nên **tăng đồng thời
không tăng thông lượng, chỉ làm phình độ trễ**. Giảm độ phân giải thì tăng thông
lượng gấp đôi, đúng như phần đo ngoại tuyến dự đoán.

### Kiểm tra tỉnh táo trên DocVQA

Nếu pipeline có lỗi hệ thống — prompt, cách chấm, hay tiền xử lý — thì mọi kết quả
ở trên đều đáng ngờ. Phép kiểm: chạy đúng thước đo **ANLS** của DocVQA rồi đối
chiếu với con số nhóm tác giả công bố.

| | ANLS |
|---|---:|
| Của dự án này (100 mẫu, split validation) | **73,8** |
| Nhóm SmolVLM công bố (toàn bộ split test) | 81,6 |

Chênh 7,8 điểm là hợp lý với chênh lệch về split và số mẫu, nên **không có lỗi hệ
thống**. Đây là phép kiểm mà một dự án đo đạc bắt buộc phải có: nó xác nhận rằng
cả bộ máy đang đo đúng thứ cần đo.

### Ngôn ngữ của chỉ dẫn có ảnh hưởng không

Chỉ dẫn trong dự án viết bằng tiếng Việt, trong khi cả hai bộ dữ liệu đều tiếng
Anh. Đây là một nghi ngờ hợp lý, nên tôi đo thay vì đoán:

| Bộ dữ liệu | Chỉ dẫn tiếng Việt | Chỉ dẫn tiếng Anh | Kết luận |
|---|---:|---:|---|
| ChartQA (200 mẫu, so theo cặp) | 67,0% | 66,0% | p = 0,80 — không phân biệt được |
| DocVQA (100 mẫu, so theo cặp) | 73,8 ANLS | 76,7 ANLS | p = 0,27 — không phân biệt được |

Chênh lệch 2,9 điểm trên DocVQA nhìn thì đáng kể, nhưng chỉ dựa trên **13 mẫu bất
đồng**, nên không đứng vững. Ngôn ngữ chỉ dẫn không phải là biến quan trọng ở đây.

---

## Sáu quy tắc đo, cài thẳng vào mã

Đo hiệu năng trên GPU laptop rất dễ ra kết quả sai theo hướng có lợi cho mình. Các
quy tắc sau nằm trong `bench/harness.py`, không phải ghi chú trong tài liệu:

1. **Vùng bấm giờ bao trọn đường đi thật**, ở mọi cấu hình được so sánh
2. **Các vòng đo là bản lặp trên cùng tập mẫu**, để so sánh được theo cặp
3. **Xen kẽ và ngẫu nhiên hoá thứ tự** các cấu hình, có seed để tái lập
4. **Báo cáo trung vị và khoảng tứ phân vị**, không dùng trung bình ± độ lệch chuẩn
5. **Chỉ tuyên bố cải thiện khi vượt ba lần nhiễu đã đo** — ở máy này là 25,5%
6. **Ghi kèm đại lượng bất biến và trạng thái máy** (token ảnh, xung nhịp, nhiệt độ)

## Bốn lỗi đo đạc đã gặp

Ghi lại vì cả bốn đều **làm kết quả đẹp lên**, tức là loại lỗi không tự lộ ra.
Notebook 01 phân tích từng lỗi bằng chính dữ liệu đã đo.

| Lỗi | Biểu hiện | Hậu quả nếu không phát hiện |
|---|---|---|
| Bộ mã hoá thị giác nằm ngoài vùng bấm giờ ở đường tối ưu | báo **3,47×** | con số thật là **1,12×**, thổi phồng gấp ba |
| Mỗi vòng đo dùng một nhóm mẫu khác nhau | IQR 86,5%, "trôi −44%" | dao động do ảnh to nhỏ bị đọc nhầm thành dao động hệ thống |
| Kết luận "không khác biệt" với 100 mẫu | p = 0,18 | với 300 mẫu, cùng hiệu ứng đó cho p = 0,002 — **kết luận đảo ngược** |
| Quên truyền `--model`, chạy nhầm mô hình 256M | độ chính xác 23%, token ảnh 640 | suýt kết luận rằng lượng tử hoá 4-bit phá hỏng mô hình |

Lỗi thứ tư còn để lại một bài học riêng: **số token ảnh là đại lượng bất biến** với
kiểu số. Khi nó đổi, tức là mình đang đo một thứ khác với thứ mình nghĩ.

---

## Cấu trúc

```
bench/
  latency_probe.py     đo độ trễ và ĐỘ NHIỄU của chính phép đo
  breakdown.py         bóc tách: thị giác · connector · prefill · sinh chữ
  preprocess_cost.py   chi phí tiền xử lý ảnh phía CPU
  prune.py             cắt token ảnh sau connector (4 cách chọn)
  quantize.py          bf16/fp16/int8/nf4 kèm kiểm tra tương đương logit
  harness.py           bộ đo chính: độ chính xác + độ trễ, nhiều cấu hình
  metrics.py           relaxed accuracy · khoảng tin cậy Wilson · McNemar
  analyze.py           phân tích theo cặp trong một lần chạy
  compare_runs.py      so sánh theo cặp giữa hai lần chạy khác nhau
  plot.py              biểu đồ đánh đổi (hai chế độ sáng/tối)
serve/
  vlm.proto            giao diện gRPC
  server.py            máy chủ suy luận, có hàng đợi và suy giảm có kiểm soát
  client_bench.py      đo độ trễ đầu-cuối dưới các mức đồng thời
notebooks/
  00_...ipynb          vì sao một ảnh tốn hơn nghìn token
  01_...ipynb          bố trí thí nghiệm cho công bằng, bốn lỗi đã gặp
tests/                 36 test cho phần lõi, chạy trên CPU trong 1 giây
results/               JSON kết quả + biểu đồ
speedrun.sh            một lệnh chạy lại tất cả
Dockerfile             đóng gói máy chủ suy luận
```

## Giới hạn

- Một mô hình, một benchmark, một GPU. Chưa khẳng định cho cấu hình khác.
- GPU laptop không khoá được xung nhịp (`nvidia-smi` báo đang bị giới hạn công suất
  và nhiệt), nên nhiễu nền khoảng 8,5% là không tránh được.
- Nhóm tác giả SmolVLM không công bố điểm ChartQA, nên chưa đối chiếu được với một
  con số độc lập. Đây là phép kiểm tra tỉnh táo còn thiếu.
- Phần cắt token ảnh mới thử bốn cách chọn đơn giản; chưa thử cách dựa trên điểm
  attention như FastV, vốn cần can thiệp sâu hơn vào vòng lặp của mô hình.

## Nguồn tham khảo

- **SmolVLM** (Hugging Face) — mô hình dùng trong thí nghiệm
- **ChartQA** — bộ dữ liệu và chuẩn chấm *relaxed accuracy*
- **FastV**, **ToMe** — các kỹ thuật cắt và gộp token ảnh
- **bitsandbytes** — lượng tử hoá int8 và nf4
