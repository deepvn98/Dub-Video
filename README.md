# Dub Sync

Ứng dụng Python cho Windows để căn MP3 tiếng Tây Ban Nha theo SRT tiếng Anh, sử dụng bản dịch đã dùng để tạo giọng đọc.

### Ý nghĩa thiết lập

- **Chia đoạn:** các block SRT chạm nhau về thời gian được gộp thành một cụm. Cụm được đối chiếu với kịch bản Anh gốc để lấy đầy đủ một hoặc nhiều dòng Spanish 1:1 tương ứng.
- **Dấu câu và xuống dòng:** không được dùng để quyết định ranh giới bản dịch.
- **Điểm bắt đầu:** chương trình tìm vị trí bắt đầu của toàn bộ phần Spanish trong MP3 rồi đặt một clip duy nhất tại START của block SRT đầu tiên trong cụm.
- **Điểm kết thúc:** END của SRT không dùng để cắt. Mép cắt nguồn được xác định từ Spanish MP3 và transcript, nằm trong khoảng giữa hai từ hoàn chỉnh. Chỉ audio tương ứng với dòng Spanish đã được xác định là thừa so với SRT mới không được đưa vào file xuất.
- **Mô hình:** `Small` nhanh và nhẹ, phù hợp đa số máy; `Medium` nhận dạng kỹ hơn nhưng chậm hơn. `Small` là mặc định.
- **Cho phép tải mô hình:** cho phép tải model faster-whisper và model đối chiếu Anh–Tây Ban Nha còn thiếu. Cả hai đều chạy local sau khi tải.

Chương trình lưu model Whisper, model tham chiếu và model đối chiếu ngữ nghĩa trên máy; kết quả nhận dạng của từng MP3 không được lưu cache. Vì vậy mỗi MP3 luôn được nhận dạng trực tiếp, còn các model không phải tải lại.

## Mở ứng dụng

Trên mỗi máy mới, cài Python 3.10 trở lên và FFmpeg/ffprobe trong `PATH`, rồi nhấp đúp **setup.bat** một lần để tạo môi trường `.venv` riêng và cài đúng thư viện cho máy đó. Sau đó nhấp đúp **run.bat**; nếu môi trường thiếu, hỏng hoặc thuộc máy khác, `run.bat` sẽ tự gọi lại trình cài.

Không sao chép `.venv` hoặc `.dub_cache` khi đóng gói gửi sang máy khác. Đây là dữ liệu phụ thuộc từng máy và sẽ được tạo lại cục bộ.

Cũng có thể chạy thủ công:

```powershell
.venv\Scripts\python.exe main.py
```

Các thư viện PySide6, faster-whisper, transformers và PyTorch được cài riêng trong `.venv` trên từng máy. Model sẽ được tải trong lần phân tích đầu tiên khi bật **Cho phép tải mô hình**.

Trên máy khác, cần Python 3.10 trở lên, FFmpeg và ffprobe trong PATH; chạy `setup.bat` để cài thư viện. Lần phân tích đầu tiên, bật **Nâng cao → Cho phép tải mô hình** nếu model chưa có trên máy. Whisper được lưu ở `.dub_cache/models`; các model đối chiếu được lưu ở `.dub_cache/alignment` và `.dub_cache/semantic`. Âm thanh và văn bản được xử lý cục bộ bằng CPU.

## Sử dụng

1. Chọn MP3/WAV Spanish, SRT tiếng Anh, nhập **Kịch bản Anh gốc** và **Bản Spanish gốc**. Cả bốn đầu vào đều bắt buộc. Mỗi dòng Spanish phải tương ứng 1:1 với dòng Anh cùng số thứ tự; số dòng của hai bản phải bằng nhau.
2. Trong **Nâng cao**, chọn `Small` hoặc `Medium`; sau đó bấm **Phân tích và đồng bộ**.
3. Chương trình gộp các block SRT liên tiếp về thời gian, căn chỉnh nội dung của từng cụm với các dòng Anh theo nội dung và thứ tự, rồi lấy đầy đủ các dòng Spanish tương ứng. Số thứ tự SRT không được coi là số thứ tự dòng kịch bản.
4. Câu dịch thừa được tự động loại theo nguyên tắc SRT là chuẩn và hiển thị trong **Nội dung tự động loại bỏ**. Nếu SRT thiếu bản dịch, bảng ghi rõ vị trí và chức năng dựng/xuất bị khóa cho đến khi người dùng bổ sung.
5. Chọn hàng → **Nghe cụm lồng tiếng**. Bảng hiển thị dòng Anh gốc, khoảng nguồn **Cắt MP3 từ/đến** và mốc **Bắt đầu đích (SRT)**; có thể sửa hai mép cắt nếu cần. Các hàng màu vàng cần nghe kiểm tra; đánh dấu **Đã kiểm tra** sau khi kiểm tra thực tế.
6. Bấm **Nghe bản dựng** để nghe toàn bộ timeline. Mỗi cụm Spanish dùng START của block SRT đầu tiên trong cụm; các mốc START ở giữa cụm không tạo điểm cắt. END của SRT không dùng để cắt Spanish. Độ dài lấy từ Spanish MP3 và transcript. Nếu cụm trước chưa nói xong ở mốc kế tiếp, cụm sau được đẩy lùi vừa đủ để không cắt từ hoặc chồng giọng.
7. Chọn **Xuất WAV / MP3**. Có thể xuất thêm SRT bản dịch theo kết quả dựng.
8. **Lưu dự án** để giữ hai kịch bản song song, các đoạn đã chia, điểm bắt đầu, trạng thái kiểm tra và model. Âm thanh không được nhúng trong dự án; khi mở lại, ứng dụng kiểm tra SHA-256 để bảo đảm đúng file.

## Phạm vi và giới hạn của bản đầu

- Đã triển khai **faster-whisper lấy mốc từ + đối chiếu từ theo thứ tự với bản dịch**. Chưa tích hợp WhisperX/forced alignment âm vị. Mốc từ là ước lượng, cần nghe kiểm tra khi điểm cắt sát lời.
- Model local chỉ tạo câu tham chiếu để xác định nghĩa và ranh giới; không gọi API dịch, không sửa bản dịch người dùng và không tạo nội dung đầu ra. Khác biệt văn phong hoặc lỗi dịch có thể làm block bị đánh dấu để kiểm tra.
- Dòng Spanish tương ứng với dòng Anh không xuất hiện trong SRT được lưu ở khu nội dung bị loại và không đi vào file dựng. Cụm SRT thiếu bản dịch được giữ thành hàng cảnh báo; dựng/xuất chỉ mở khi mọi cụm đã có bản dịch.
- Mỗi lần phân tích, MP3 được nhận dạng lại; chương trình không lưu cache kết quả nhận dạng. Khi thay đầu vào, bảng cũ được giữ để người dùng vẫn có thể lưu dự án; chức năng nghe/xuất bị khóa đến khi phân tích lại.
- “Khớp từ” là tỷ lệ từ bản dịch được ghép chính xác với bản nhận dạng, không phải xác suất đúng. Số đo, cách viết khác và lỗi nhận dạng có thể bị đánh dấu dù lời đọc đúng.
- Mới hỗ trợ luồng thuyết minh tuần tự, SRT không chồng thời gian. Không có chức năng khớp khẩu hình, tạo lại giọng, tách nhạc hay ghép video trong bản này.
- Điểm đầu/cuối và đoạn MP3 thừa dựa trên mốc từ của faster-whisper. Sai số nhận dạng có thể làm ranh giới lệch nhẹ, vì vậy các hàng có mức khớp thấp vẫn cần nghe kiểm tra.
- Nút Hủy dừng giữa các đoạn nhận dạng hoặc dừng FFmpeg đang chạy. Đang nạp/tải mô hình có thể chưa dừng ngay; giao diện vẫn phản hồi.

## Kiểm thử

```powershell
python -m unittest discover -s tests -p "test_core.py" -v
python -X utf8 tests/smoke_ui.py
```

Kiểm thử lõi bao gồm gộp block SRT liên tiếp, giữ riêng block không liên tiếp, ánh xạ một cụm qua nhiều dòng Anh–Spanish, phát hiện dòng thiếu/thừa, ghép từ bị thiếu/lặp, SRT lỗi, điểm bắt đầu tăng dần, cổng kiểm tra trước xuất, hủy không ghi đè kết quả và dựng WAV không cắt từ hoặc chồng tiếng. Kiểm thử UI dùng dữ liệu nhận dạng và ánh xạ mẫu cục bộ.
