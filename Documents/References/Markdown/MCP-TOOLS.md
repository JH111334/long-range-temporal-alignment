# MCP và công cụ của project

## Quyết định

Project chỉ bật `filesystem`, `memory_keeper` và `playwright` trong `.codex/config.toml`:

- `filesystem`: đọc và kiểm tra artifact trong phạm vi repository.
- `memory_keeper`: giữ quyết định theo đúng project và chủ đề.
- `playwright`: kiểm tra giao diện Streamlit bằng trình duyệt khi UI đã chạy.

Các MCP `docker`, `github`, `powerpoint`, `socialcrawl`, `winston_ai` và `youtube_knowledge` bị tắt ở cấp project vì chưa thuộc đường găng nghiên cứu. Không sửa cấu hình toàn cục để tránh làm hỏng workflow AIC hoặc project khác. GitHub được thao tác bằng `git`/`gh` có kiểm tra remote; nghiên cứu nguồn dùng web và nguồn chính thức.

Không cần MCP riêng cho Streamlit. Unit/UI contract dùng `st.testing.v1.AppTest`; browser smoke dùng Playwright. Cấu hình project chỉ có hiệu lực khi project được tin cậy và phiên Codex mới được khởi động.

## Hoàn tác

Đặt lại `enabled = true` cho MCP cần dùng hoặc xóa đúng bảng tương ứng khỏi `.codex/config.toml`, sau đó khởi động phiên mới. Không xóa token hay cấu hình máy khi chỉ cần thay đổi phạm vi project.

Nguồn: [Codex configuration reference](https://developers.openai.com/codex/config-reference), [Streamlit AppTest](https://docs.streamlit.io/develop/concepts/app-testing).
