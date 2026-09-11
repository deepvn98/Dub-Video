from __future__ import annotations

import copy
import sys
import tempfile
import threading
import uuid
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .core import (
    Cancelled,
    Project,
    align_rows,
    audio_duration,
    audio_for_extra_translations,
    extract_extra_translations,
    fingerprint,
    alignment_references,
    load_alignment_model,
    load_project,
    parallel_script_rows,
    parse_parallel_lines,
    parse_srt,
    plan,
    read_text,
    render,
    render_clip,
    save_project,
    structural_issue_counts,
    timestamp,
    transcribe,
    translated_srt,
)

ROOT = Path(__file__).resolve().parent.parent


class FileInput(QLineEdit):
    def __init__(self, placeholder):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.setText(urls[0].toLocalFile())
            event.acceptProposedAction()


class Worker(QThread):
    progress = Signal(int, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, function):
        super().__init__()
        self.function = function
        self.cancel = threading.Event()

    def run(self):
        try:
            result = self.function(self.cancel, self.progress.emit)
            self.succeeded.emit(result)
        except Cancelled:
            self.failed.emit('Đã hủy tác vụ. File đầu vào không thay đổi.')
        except Exception as exc:
            self.failed.emit(str(exc))


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Dub Sync · Đồng bộ giọng đọc')
        self.resize(1280, 880)
        self.setMinimumSize(1000, 720)
        self.project = None
        self.inputs_changed = False
        self.embedded_cues = None
        self.worker = None
        self.filling = False
        self.dirty = False
        self.last_result = None
        self.preview_dir = tempfile.TemporaryDirectory(prefix='dub-sync-preview-', ignore_cleanup_errors=True)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(.85)
        self.player.setAudioOutput(self.audio_output)
        self.player.errorOccurred.connect(lambda error, message: self.status.setText('Không phát được âm thanh: ' + message))

        base = QWidget()
        self.setCentralWidget(base)
        layout = QVBoxLayout(base)
        layout.setContentsMargins(22, 16, 22, 16)
        title = QLabel('Dub Sync')
        title.setObjectName('title')
        layout.addWidget(title)
        layout.addWidget(QLabel('Đưa giọng đọc đã dịch về đúng nhịp của phụ đề gốc.'))

        toolbar = QHBoxLayout()
        self.open_button = self.button('Mở dự án', self.open_project, toolbar)
        self.save_button = self.button('Lưu dự án', self.save, toolbar)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.inputs = QGroupBox('1 · Nạp dữ liệu')
        inputs_layout = QFormLayout(self.inputs)
        self.audio_path = FileInput('Kéo thả MP3/WAV giọng đã dịch hoặc bấm Chọn file')
        self.srt_path = FileInput('Kéo thả SRT tiếng Anh')
        for label, field, filt in [('Âm thanh lồng tiếng', self.audio_path, 'Âm thanh (*.mp3 *.wav *.m4a *.flac *.ogg)'),
                                   ('Phụ đề gốc', self.srt_path, 'Phụ đề (*.srt)')]:
            row = QHBoxLayout()
            row.addWidget(field)
            self.button('Chọn file', lambda checked=False, f=field, t=filt: self.choose(f, t), row)
            inputs_layout.addRow(label, row)
            field.textChanged.connect(self.invalidate)
        self.source_script = QPlainTextEdit()
        self.source_script.setPlaceholderText(
            'Bắt buộc: mỗi dòng là một câu/đơn vị tiếng Anh gốc.')
        self.source_script.setMaximumHeight(72)
        self.source_script.textChanged.connect(self.invalidate)
        inputs_layout.addRow('Kịch bản Anh gốc', self.source_script)
        source_bar = QHBoxLayout()
        self.button('Nhập TXT', self.import_source_script, source_bar)
        source_bar.addWidget(QLabel(
            'Mỗi dòng phải tương ứng 1:1 với một dòng trong bản Spanish.'))
        source_bar.addStretch()
        inputs_layout.addRow('', source_bar)
        self.translation = QPlainTextEdit()
        self.translation.setPlaceholderText(
            'Bắt buộc: mỗi dòng Spanish tương ứng 1:1 với dòng kịch bản Anh cùng số thứ tự.')
        self.translation.setMaximumHeight(90)
        self.translation.textChanged.connect(self.invalidate)
        inputs_layout.addRow('Bản dịch', self.translation)
        translation_bar = QHBoxLayout()
        self.button('Nhập TXT', self.import_translation, translation_bar)
        translation_bar.addWidget(QLabel(
            'Giữ đúng số dòng và thứ tự của kịch bản Anh gốc.'))
        translation_bar.addStretch()
        inputs_layout.addRow('', translation_bar)
        layout.addWidget(self.inputs)

        settings = QHBoxLayout()
        self.advanced_toggle = QCheckBox('Nâng cao')
        settings.addWidget(self.advanced_toggle)
        settings.addStretch()
        self.analyze_button = self.button('Phân tích và đồng bộ', self.analyze, settings)
        self.analyze_button.setObjectName('primary')
        layout.addLayout(settings)

        self.advanced = QGroupBox('Thiết lập nhận dạng')
        advanced = QFormLayout(self.advanced)
        sync_settings = QHBoxLayout()
        self.model = QComboBox()
        self.model.addItem('Small — nhanh', 'small')
        self.model.addItem('Medium — cân bằng', 'medium')
        self.model.setToolTip('Small chạy nhanh và nhẹ; Medium nhận dạng kỹ hơn nhưng chậm hơn. Cả hai đều chạy bằng CPU.')
        self.model.currentIndexChanged.connect(self.recognition_settings_changed)
        self.download = QCheckBox('Cho phép tải mô hình')
        self.download.setToolTip('Cho phép tải model nhận dạng và model đối chiếu Anh–Tây Ban Nha còn thiếu. Dữ liệu dự án không được gửi lên dịch vụ.')
        sync_settings.addWidget(QLabel('Mô hình'))
        sync_settings.addWidget(self.model)
        sync_settings.addWidget(self.download)
        sync_settings.addStretch()
        advanced.addRow('Nhận dạng', sync_settings)
        advanced_help = QLabel(
            'SRT là chuẩn thời gian. Các block chạm nhau về thời gian được gộp thành một cụm; chương trình đối chiếu cụm với dòng Anh gốc và lấy đầy đủ dòng Spanish 1:1 tương ứng. '
            'Sau khi hoàn tất ánh xạ văn bản, chương trình mới nhận dạng và cắt một đoạn MP3 cho mỗi cụm. Chỉ START của block đầu tiên được dùng để đặt audio; END SRT không dùng để cắt. '
            'Bản dịch người dùng không bị sửa và dấu câu không quyết định ranh giới. Lời dịch và MP3 không thuộc SRT sẽ bị loại.'
        )
        advanced_help.setWordWrap(True)
        advanced.addRow('Giải thích', advanced_help)
        self.advanced.hide()
        self.advanced_toggle.toggled.connect(self.advanced.setVisible)
        layout.addWidget(self.advanced)

        result_bar = QHBoxLayout()
        self.summary = QLabel('2 · Kết quả — chưa có dữ liệu')
        result_bar.addWidget(self.summary)
        result_bar.addStretch()
        self.only_issues = QCheckBox('Chỉ hiện đoạn cần kiểm tra')
        self.only_issues.toggled.connect(self.filter_rows)
        result_bar.addWidget(self.only_issues)
        layout.addLayout(result_bar)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            'Dòng SRT', 'Dòng Anh gốc', 'Nội dung SRT', 'Bản dịch',
            'Cắt MP3 từ (giây)', 'Cắt MP3 đến (giây)',
            'Bắt đầu đích (SRT)', 'Trạng thái', 'Đã kiểm tra'
        ])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setWordWrap(True)
        self.table.setAlternatingRowColors(True)
        self.table.itemChanged.connect(self.table_changed)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for i, width in enumerate([70, 90, 220, 245, 115, 115, 125, 195, 80]):
            self.table.setColumnWidth(i, width)
        layout.addWidget(self.table, 1)
        hint = QLabel(
            'Mỗi hàng là một cụm gồm các block SRT liên tiếp về thời gian. Cụm nhận đầy đủ phần Spanish tương ứng '
            'và dùng START của block đầu tiên để đặt một đoạn audio hoàn chỉnh.')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        discarded_box = QGroupBox('3 · Nội dung tự động loại bỏ — SRT là chuẩn')
        discarded_layout = QHBoxLayout(discarded_box)
        translation_discard_column = QVBoxLayout()
        translation_discard_column.addWidget(QLabel('Bản dịch thừa'))
        self.discarded_translation = QPlainTextEdit()
        self.discarded_translation.setReadOnly(True)
        self.discarded_translation.setMaximumHeight(48)
        self.discarded_translation.setPlaceholderText('Không có câu dịch thừa.')
        translation_discard_column.addWidget(self.discarded_translation)
        discarded_layout.addLayout(translation_discard_column, 1)
        audio_discard_column = QVBoxLayout()
        audio_discard_column.addWidget(QLabel('Đoạn MP3 thừa'))
        self.discarded_audio = QPlainTextEdit()
        self.discarded_audio.setReadOnly(True)
        self.discarded_audio.setMaximumHeight(48)
        self.discarded_audio.setPlaceholderText('Không có đoạn lời MP3 thừa được phát hiện.')
        self.discarded_audio.setToolTip(
            'Chỉ gồm audio tương ứng với phần bản dịch đã xác định là thừa so với SRT; lỗi nhận dạng từ không tự làm mất audio.')
        audio_discard_column.addWidget(self.discarded_audio)
        discarded_layout.addLayout(audio_discard_column, 1)
        layout.addWidget(discarded_box)

        actions = QHBoxLayout()
        self.source_button = self.button('Nghe cụm lồng tiếng', self.preview_source, actions)
        self.preview_button = self.button('Nghe bản dựng', self.preview, actions)
        self.export_button = self.button('Xuất WAV / MP3', self.export, actions)
        self.export_button.setObjectName('primary')
        self.srt_button = self.button('Xuất SRT bản dịch', self.export_srt, actions)
        actions.addStretch()
        self.cancel_button = self.button('Hủy xử lý', self.cancel, actions)
        self.cancel_button.setEnabled(False)
        layout.addLayout(actions)

        playback = QHBoxLayout()
        self.button('Phát / Tạm dừng', self.toggle_play, playback)
        self.button('Dừng', self.player.stop, playback)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.sliderMoved.connect(self.player.setPosition)
        self.player.durationChanged.connect(lambda ms: self.seek.setRange(0, ms))
        self.player.positionChanged.connect(self.play_position)
        playback.addWidget(self.seek, 1)
        self.play_time = QLabel('00:00:00 / 00:00:00')
        playback.addWidget(self.play_time)
        layout.addLayout(playback)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel('Sẵn sàng. Nhận dạng và dựng âm thanh chạy trên máy của bạn.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.setStyleSheet('''
            QMainWindow { background: #f4f6fa; }
            QWidget { font-family: "Segoe UI"; font-size: 12px; color: #17283f; }
            QLabel#title { font-size: 27px; font-weight: 700; color: #145d7a; }
            QGroupBox { background: white; border: 1px solid #d8e1e9; border-radius: 7px; margin-top: 9px; padding: 13px 8px 7px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; font-weight: 600; }
            QPushButton { padding: 7px 12px; border: 1px solid #b8c9d8; border-radius: 5px; background: white; }
            QPushButton:hover { background: #e6f0f7; }
            QPushButton#primary { background: #146e87; color: white; border: none; font-weight: 600; }
            QPushButton:disabled { background: #e4e9ee; color: #81909d; }
            QLineEdit, QPlainTextEdit, QComboBox { background: white; border: 1px solid #bbcbd8; border-radius: 4px; padding: 4px; }
            QTableWidget { background: white; alternate-background-color: #f3f7fa; gridline-color: #e1e7ed; selection-background-color: #cee8f2; selection-color: #17283f; }
            QHeaderView::section { background: #e9eff5; padding: 7px; border: none; font-weight: 600; }
            QProgressBar { border: none; background: #e0e7ee; height: 12px; text-align: center; border-radius: 4px; }
            QProgressBar::chunk { background: #2385a0; border-radius: 4px; }
        ''')
        self.update_enabled()

    def button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def error(self, message):
        QMessageBox.warning(self, 'Dub Sync', str(message))

    def choose(self, field, filt):
        path, _ = QFileDialog.getOpenFileName(self, 'Chọn file', '', filt)
        if path:
            field.setText(path)

    def import_translation(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Chọn bản dịch', '', 'Văn bản (*.txt *.md)')
        if path:
            try:
                self.translation.setPlainText(read_text(path))
            except Exception as exc:
                self.error(exc)

    def import_source_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Chọn kịch bản tiếng Anh gốc', '', 'Văn bản (*.txt *.md)')
        if path:
            try:
                self.source_script.setPlainText(read_text(path))
            except Exception as exc:
                self.error(exc)

    def invalidate(self):
        if self.filling:
            return
        self.inputs_changed = True
        self.summary.setText('2 · Đầu vào đã đổi — bảng bên dưới là kết quả trước; phân tích lại để cập nhật')
        self.update_enabled()

    def update_enabled(self):
        busy = self.worker is not None
        for widget in [self.inputs, self.open_button, self.advanced]:
            widget.setEnabled(not busy)
        has_srt = bool(self.srt_path.text().strip() or self.embedded_cues)
        has_inputs = bool(self.audio_path.text().strip() and has_srt
                          and self.source_script.toPlainText().strip()
                          and self.translation.toPlainText().strip())
        self.analyze_button.setEnabled(not busy and has_inputs)
        editable = not busy and self.project is not None and not self.inputs_changed
        self.table.setEnabled(editable)
        structurally_ready = (editable and structural_issue_counts(self.project.rows) == (0, 0))
        for widget in [self.source_button, self.preview_button, self.export_button]:
            widget.setEnabled(structurally_ready)
        self.save_button.setEnabled(not busy and self.project is not None)
        self.srt_button.setEnabled(structurally_ready and self.last_result is not None)
        self.cancel_button.setEnabled(busy)

    def start_task(self, function, callback):
        if self.worker:
            return
        self.player.stop()
        self.player.setSource(QUrl())
        self.progress.setValue(0)
        worker = Worker(function)
        self.worker = worker
        worker.progress.connect(self.on_progress)
        worker.succeeded.connect(callback)
        worker.failed.connect(self.task_error)
        worker.finished.connect(self.task_finished)
        self.update_enabled()
        worker.start()

    def on_progress(self, value, message):
        self.progress.setValue(value)
        self.status.setText(message)

    def task_error(self, message):
        self.status.setText(message)
        if not message.startswith('Đã hủy'):
            self.error(message)

    def task_finished(self):
        worker = self.worker
        self.worker = None
        worker.deleteLater()
        self.update_enabled()

    def cancel(self):
        if self.worker:
            self.worker.cancel.set()
            self.status.setText('Đang hủy… Nhận dạng sẽ dừng sau đoạn đang xử lý; nạp mô hình có thể cần thêm thời gian.')

    def analyze(self):
        if not self.discard_ok():
            return
        try:
            audio = str(Path(self.audio_path.text().strip()).resolve())
            if not Path(audio).is_file():
                raise ValueError('Hãy chọn MP3/WAV giọng đã dịch hợp lệ.')
            srt_path = self.srt_path.text().strip()
            if srt_path:
                cues = parse_srt(read_text(srt_path))
            elif self.embedded_cues:
                cues = copy.deepcopy(self.embedded_cues)
            else:
                raise ValueError('Hãy chọn file SRT hợp lệ.')
            source_script = self.source_script.toPlainText().strip()
            if not source_script:
                raise ValueError('Hãy nhập kịch bản tiếng Anh gốc.')
            english_lines = parse_parallel_lines(source_script, 'Kịch bản tiếng Anh gốc')
            spanish_script = self.translation.toPlainText().strip()
            spanish_lines = parse_parallel_lines(spanish_script, 'Bản Spanish gốc')
            if len(english_lines) != len(spanish_lines):
                raise ValueError(
                    f'Kịch bản Anh có {len(english_lines)} dòng nhưng bản Spanish có '
                    f'{len(spanish_lines)} dòng. Hai bản phải tương ứng 1:1.')
            model, download = self.model.currentData(), self.download.isChecked()
        except Exception as exc:
            self.error(exc)
            return

        def work(cancel, progress):
            duration = audio_duration(audio, cancel)
            progress(3, 'Đang nạp mô hình đối chiếu Anh–Tây Ban Nha…')
            alignment_model = load_alignment_model(ROOT / '.dub_cache', allow_download=download)
            progress(4, 'Đang tạo tham chiếu để ghép các block SRT liên tiếp với Spanish…')
            references = alignment_references(
                cues, alignment_model, cancel=cancel, progress=progress)
            proposed = parallel_script_rows(
                cues, english_lines, spanish_lines, references,
                alignment_model[2], cancel=cancel, progress=progress)
            rows, discarded_translations = extract_extra_translations(proposed)
            _, missing = structural_issue_counts(rows)
            if missing:
                progress(90, 'Có cụm SRT thiếu bản dịch; bỏ qua nhận dạng MP3 cho đến khi được bổ sung.')
                return Project(
                    audio, fingerprint(audio, cancel), duration, cues, rows,
                    model, discarded_translations,
                    source_script=source_script,
                    spanish_script=spanish_script)
            progress(
                31,
                f'Đã chia đủ {len(rows)} cụm Tây Ban Nha theo các nhóm SRT liên tiếp; '
                'bắt đầu nhận dạng MP3 để xác định mép cắt.'
            )
            segments, digest = transcribe(audio, ROOT / '.dub_cache', model=model, allow_download=download, cancel=cancel, progress=progress)
            progress(90, 'Đang đối chiếu bản dịch và mốc từ…')
            align_rows(proposed, segments, duration)
            discarded_audio = audio_for_extra_translations(proposed, duration)
            discarded_audio.sort(key=lambda item: item.start)
            return Project(
                audio, digest, duration, cues, rows, model,
                discarded_translations, discarded_audio, source_script,
                spanish_script)
        self.start_task(work, self.set_project)

    def set_project(self, project):
        self.project = project
        self.inputs_changed = False
        self.last_result = None
        self.dirty = True
        self.filling = True
        self.model.setCurrentIndex(max(0, self.model.findData(project.model)))
        self.filling = False
        self.fill_table()
        self.progress.setValue(100)
        _, missing = structural_issue_counts(project.rows)
        if missing:
            message = (f'Có {missing} cụm SRT thiếu bản dịch. Đã giữ riêng {len(project.discarded_translations)} dòng dịch thừa. '
                       'Hãy bổ sung bản dịch rồi phân tích lại; chương trình chưa cho dựng hoặc xuất.')
            self.status.setText(message)
            QMessageBox.warning(self, 'Bản dịch và SRT chưa khớp', message)
        else:
            self.status.setText(
                f'Đã dùng SRT làm chuẩn; loại {len(project.discarded_translations)} câu dịch và '
                f'{len(project.discarded_audio)} đoạn lời MP3 thừa. '
                'Hãy xem khu nội dung bị loại và nghe các hàng cảnh báo.'
            )
        self.update_enabled()

    def fill_table(self):
        self.filling = True
        p = self.project
        self.table.setRowCount(len(p.rows))
        for i, row in enumerate(p.rows):
            mapped = 0 <= row.first <= row.last < len(p.cues)
            if mapped:
                selected = p.cues[row.first:row.last + 1]
                cue_range = (str(selected[0].number) if len(selected) == 1
                             else f'{selected[0].number}–{selected[-1].number}')
                source_text = ' '.join(cue.text for cue in selected)
                target_start = timestamp(selected[0].start)
            else:
                cue_range, source_text, target_start = '—', '— Không có cụm SRT tương ứng —', '—'
            status = f'Khớp từ {row.confidence:.0%}' if row.translation and row.confidence else 'Chưa thể đối chiếu'
            if row.issues:
                status += '\n' + '; '.join(row.issues)
            translation = row.translation or '— Thiếu bản dịch —'
            has_audio_range = row.translation and row.confidence and row.source_end > row.source_start
            source_start = f'{row.source_start:.3f}' if has_audio_range else '—'
            source_end = f'{row.source_end:.3f}' if has_audio_range else '—'
            if row.source_line < 0:
                source_line = '—'
            elif row.source_line_last > row.source_line:
                source_line = f'{row.source_line + 1}–{row.source_line_last + 1}'
            else:
                source_line = str(row.source_line + 1)
            values = [cue_range, source_line, source_text, translation,
                      source_start, source_end, target_start, status]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col not in (4, 5) or not has_audio_range:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(value)
                if col == 7 and row.issues and not row.reviewed:
                    item.setBackground(QColor('#fff0ce'))
                self.table.setItem(i, col, item)
            checked = QTableWidgetItem()
            checked.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
            checked.setCheckState(Qt.CheckState.Checked if row.reviewed else Qt.CheckState.Unchecked)
            self.table.setItem(i, 8, checked)
            self.table.setRowHeight(i, 70)
        _, missing = structural_issue_counts(p.rows)
        mapped_segments = sum(row.first >= 0 for row in p.rows)
        mismatch = f' · Thiếu {missing} đoạn dịch' if missing else ' · Đủ bản dịch'
        removed = f' · Đã loại {len(p.discarded_translations)} câu dịch, {len(p.discarded_audio)} đoạn MP3'
        self.summary.setText(f'2 · {mapped_segments} cụm / {len(p.cues)} block SRT{mismatch}{removed}'
                             f' · Đã đối chiếu kịch bản Anh–Spanish 1:1'
                             f' · MP3 {p.audio_duration:.1f}s → {p.cues[-1].end:.1f}s')
        self.discarded_translation.setPlainText(
            '\n'.join(f'{i}. {text}' for i, text in enumerate(p.discarded_translations, 1)))
        self.discarded_audio.setPlainText('\n'.join(
            f'{timestamp(item.start)} → {timestamp(item.end)}: {item.text}' for item in p.discarded_audio))
        self.filling = False
        self.filter_rows()

    def filter_rows(self):
        if self.project:
            for i, row in enumerate(self.project.rows):
                issue = (bool(row.issues and not row.reviewed) or row.source_end <= row.source_start
                         or row.first < 0 or not row.translation)
                self.table.setRowHidden(i, self.only_issues.isChecked() and not issue)

    def table_changed(self, item):
        if self.filling or not self.project:
            return
        row = self.project.rows[item.row()]
        try:
            if item.column() in (4, 5):
                value = float(item.text().replace(',', '.'))
                if not 0 <= value <= self.project.audio_duration:
                    raise ValueError()
                if item.column() == 4:
                    if row.source_end and value >= row.source_end:
                        raise ValueError()
                    row.source_start = value
                else:
                    if value <= row.source_start:
                        raise ValueError()
                    row.source_end = value
                row.confidence = max(row.confidence, .0001)
                row.reviewed = False
            elif item.column() == 8:
                row.reviewed = item.checkState() == Qt.CheckState.Checked
            self.dirty = True
            if item.column() != 8:
                self.last_result = None
        except ValueError:
            self.error('Mép cắt MP3 phải nằm trong thời lượng file và điểm bắt đầu phải nhỏ hơn điểm kết thúc.')
        self.fill_table()
        self.update_enabled()

    def recognition_settings_changed(self):
        if self.filling or not self.project:
            return
        self.inputs_changed = True
        self.last_result = None
        self.dirty = True
        self.summary.setText('2 · Thiết lập nhận dạng đã đổi — hãy phân tích lại')
        self.status.setText('Mô hình mới sẽ được dùng trong lần phân tích tiếp theo.')
        self.update_enabled()

    def preview_source(self):
        if not self.project:
            return
        index = self.table.currentRow()
        if index < 0:
            self.error('Chọn một hàng trong bảng để nghe cụm gốc.')
            return
        p = copy.deepcopy(self.project)
        row = p.rows[index]
        if not 0 <= row.source_start < row.source_end <= p.audio_duration:
            self.error('Hãy sửa điểm bắt đầu MP3 trước khi nghe cụm này.')
            return
        destination = Path(self.preview_dir.name) / f'source-{uuid.uuid4().hex}.wav'

        def work(cancel, progress):
            if fingerprint(p.audio, cancel) != p.audio_hash:
                raise ValueError('File âm thanh đã thay đổi; hãy phân tích lại.')
            progress(15, 'Đang chuẩn bị cụm gốc…')
            render_clip(p.audio, row, destination, cancel=cancel)
            return str(destination)
        self.start_task(work, self.play_file)

    def preview(self):
        if not self.project:
            return
        p = copy.deepcopy(self.project)
        destination = Path(self.preview_dir.name) / f'preview-{uuid.uuid4().hex}.wav'
        self.start_task(lambda cancel, progress: render(p, destination, cancel, progress, require_review=False), self.preview_done)

    def preview_done(self, result):
        self.last_result = result
        self.play_file(result['path'])
        delayed = sum(timing.get('delay', 0) > .001 for timing in result['timings'])
        self.status.setText(
            f'Đang nghe bản dựng thử. {delayed} cụm được đẩy lùi để không cắt từ hoặc chồng tiếng. '
            'Các đoạn chưa xác nhận vẫn cần kiểm tra trước khi xuất chính thức.')

    def play_file(self, path):
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player.play()
        self.progress.setValue(100)
        self.status.setText('Đang phát: ' + Path(path).name)

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def play_position(self, ms):
        if not self.seek.isSliderDown():
            self.seek.setValue(ms)
        self.play_time.setText(f'{timestamp(ms / 1000)[:8]} / {timestamp(self.player.duration() / 1000)[:8]}')

    def export(self):
        if not self.project:
            return
        try:
            plan(self.project)
        except Exception as exc:
            self.error(exc)
            return
        path, selected = QFileDialog.getSaveFileName(self, 'Xuất âm thanh', str(ROOT / 'output' / 'dong_bo.wav'), 'WAV (*.wav);;MP3 (*.mp3)')
        if not path:
            return
        if not Path(path).suffix:
            path += '.mp3' if 'MP3' in selected else '.wav'
        p = copy.deepcopy(self.project)
        self.start_task(lambda cancel, progress: render(p, path, cancel, progress), self.export_done)

    def export_done(self, result):
        self.last_result = result
        delayed = sum(timing.get('delay', 0) > .001 for timing in result['timings'])
        self.status.setText(f'Đã xuất: {result["path"]} · {delayed} cụm được đẩy lùi để tránh chồng tiếng')
        QMessageBox.information(
            self, 'Xuất thành công',
            result['path'] + f'\n{delayed} cụm được đẩy lùi để giữ trọn lời và không chồng tiếng.'
            '\nBạn có thể xuất thêm SRT bản dịch theo timing thực tế hoặc lưu dự án.')

    def export_srt(self):
        if not self.last_result or not self.project:
            return
        try:
            plan(self.project)
            path, _ = QFileDialog.getSaveFileName(self, 'Xuất SRT bản dịch', 'dong_bo.es.srt', 'Phụ đề (*.srt)')
            if path:
                if Path(path).resolve() == Path(self.project.audio).resolve():
                    raise ValueError('Không thể ghi đè âm thanh gốc.')
                Path(path).write_text(translated_srt(self.project, self.last_result['timings']), encoding='utf-8-sig')
                self.status.setText('Đã xuất SRT theo các cụm âm thanh đã dựng: ' + path)
        except Exception as exc:
            self.error(exc)

    def save(self):
        if not self.project:
            return False
        path, _ = QFileDialog.getSaveFileName(self, 'Lưu dự án', 'du_an.dubsync.json', 'Dự án (*.json)')
        if not path:
            return False
        try:
            save_project(path, self.project)
            self.dirty = False
            self.status.setText('Đã lưu dự án: ' + path)
            return True
        except Exception as exc:
            self.error(exc)
            return False

    def discard_ok(self):
        if not self.project or not self.dirty:
            return True
        answer = QMessageBox.question(self, 'Lưu thay đổi?', 'Bạn muốn lưu dự án hiện tại trước khi tiếp tục?',
                                      QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def open_project(self):
        if not self.discard_ok():
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Mở dự án', '', 'Dự án (*.json)')
        if not path:
            return
        try:
            p = load_project(path)
            if not Path(p.audio).is_file():
                replacement, _ = QFileDialog.getOpenFileName(self, 'Tìm lại âm thanh gốc', '', 'Âm thanh (*.mp3 *.wav *.m4a *.flac)')
                if not replacement:
                    return
                p.audio = str(Path(replacement).resolve())
        except Exception as exc:
            self.error(exc)
            return

        def work(cancel, progress):
            progress(20, 'Đang kiểm tra âm thanh của dự án…')
            if fingerprint(p.audio, cancel) != p.audio_hash:
                raise ValueError('Âm thanh không trùng với file đã phân tích trong dự án.')
            return p
        self.start_task(work, self.open_done)

    def open_done(self, p):
        self.embedded_cues = copy.deepcopy(p.cues)
        self.filling = True
        self.audio_path.setText(p.audio)
        self.srt_path.clear()
        self.srt_path.setPlaceholderText('Đang dùng SRT lưu trong dự án; có thể chọn SRT khác')
        self.translation.setPlainText(
            p.spanish_script or '\n'.join(r.translation for r in p.rows))
        self.source_script.setPlainText(p.source_script)
        self.filling = False
        self.set_project(p)
        self.dirty = False

    def closeEvent(self, event):
        if self.worker:
            self.cancel()
            event.ignore()
            self.status.setText('Đang hủy tác vụ. Hãy đóng cửa sổ lại sau khi tác vụ dừng.')
            return
        if not self.discard_ok():
            event.ignore()
            return
        self.player.stop()
        self.player.setSource(QUrl())
        # Release Windows multimedia file handles before cleaning preview files.
        import shiboken6
        shiboken6.delete(self.player)
        self.preview_dir.cleanup()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Dub Sync')
    window = Window()
    window.show()
    sys.exit(app.exec())
