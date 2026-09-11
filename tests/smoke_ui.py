"""Exercise the UI worker, sample alignment, edits and preview export.

Run: python -X utf8 tests/smoke_ui.py
QT_QPA_PLATFORM=offscreen makes this a hidden test, not a desktop launcher.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

import dub_sync.ui as ui
from dub_sync.core import (fingerprint, parse_parallel_lines, parse_srt, plan, read_text,
                           structural_issue_counts, tokens)


def sample_transcribe(path, cache_dir, **kwargs):
    """Keep this UI test offline without relying on a runtime ASR cache."""
    del cache_dir
    segments = json.loads((ROOT / 'analysis_spanish_transcript.json').read_text(encoding='utf-8'))
    return segments, fingerprint(path, kwargs.get('cancel'))


ui.transcribe = sample_transcribe


def contiguous_groups(cues):
    """Build parallel-script sample lines; production no longer groups SRT."""
    groups = []
    first = 0
    for index in range(1, len(cues)):
        if abs(cues[index].start - cues[index - 1].end) > .002:
            groups.append((first, index - 1))
            first = index
    groups.append((first, len(cues) - 1))
    return groups


class ConstantSemanticModel:
    def encode(self, texts, **kwargs):
        del kwargs
        return np.ones((len(texts), 1), dtype=float)


def sample_translate_references(cues):
    """Create deterministic local references for the offline UI test."""
    target_lines = parse_parallel_lines(
        (ROOT / 'samples' / 'translation_es.txt').read_text(encoding='utf-8'),
        'Spanish sample')
    groups = contiguous_groups(cues)
    references = []
    for (first, last), target_line in zip(groups, target_lines):
        target = tokens(target_line)
        weights = [max(1, len(tokens(cues[index].text)))
                   for index in range(first, last + 1)]
        total = sum(weights)
        boundaries = [0]
        cumulative = 0
        for weight in weights:
            cumulative += weight
            boundaries.append(round(len(target) * cumulative / total))
        references.extend(' '.join(target[boundaries[index]:boundaries[index + 1]])
                          for index in range(len(weights)))
    return references


ui.load_alignment_model = lambda *args, **kwargs: (None, None, ConstantSemanticModel())
ui.alignment_references = lambda cues, model, **kwargs: sample_translate_references(cues)
Window = ui.Window

app = QApplication([])
font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'segoeui.ttf'
if font_path.exists():
    QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont('Segoe UI', 10))
w = Window()
errors = []
warnings = []
w.error = lambda message: errors.append(str(message))
ui.QMessageBox.warning = lambda parent, title, message: warnings.append((title, str(message)))
w.show()
w.advanced_toggle.setChecked(True)
audio_path = os.environ.get('DUB_SYNC_TEST_AUDIO', '')
srt_path = os.environ.get('DUB_SYNC_TEST_SRT', '')
if not Path(audio_path).is_file() or not Path(srt_path).is_file():
    print('SKIP: set DUB_SYNC_TEST_AUDIO and DUB_SYNC_TEST_SRT to run the UI smoke test')
    w.close()
    raise SystemExit(0)
w.audio_path.setText(audio_path)
w.srt_path.setText(srt_path)
w.source_script.setPlainText('\n'.join(
    ' '.join(cue.text for cue in parse_srt(read_text(srt_path))[first:last + 1])
    for first, last in contiguous_groups(parse_srt(read_text(srt_path)))))
w.translation.setPlainText((ROOT / 'samples' / 'translation_es.txt').read_text(encoding='utf-8'))
w.analyze()

def wait_worker(timeout=240):
    deadline = time.monotonic() + timeout
    while w.worker is not None:
        app.processEvents()
        if time.monotonic() > deadline:
            w.cancel()
            raise AssertionError('UI worker timeout')
        time.sleep(.02)
    app.processEvents()

wait_worker()
assert not errors, errors
assert w.project is not None
assert len(w.project.cues) == 58
assert w.table.rowCount() == len(w.project.rows)
assert structural_issue_counts(w.project.rows) == (0, 0), warnings
expected_rows = len(w.project.rows)
valid_project = w.project
broken_project = copy.deepcopy(valid_project)
broken_project.discarded_translations.append('Frase adicional.')
w.set_project(broken_project)
assert 'Frase adicional.' in w.discarded_translation.toPlainText()
assert w.preview_button.isEnabled() and w.export_button.isEnabled()
w.set_project(valid_project)
w.table.selectRow(0)
previous = w.project.rows[0].source_start
w.table.item(0, 4).setText(str(previous + .01))
assert abs(w.project.rows[0].source_start - previous - .01) < .00001
w.table.item(0, 4).setText(str(previous))
w.preview()
wait_worker()
assert not errors, errors
assert w.last_result and len(w.last_result['timings']) == expected_rows
assert Path(w.last_result['path']).exists()
w.player.stop()
w.resize(1400, 960)
app.processEvents()
w.grab().save(str(ROOT / 'output' / 'ui_sample.png'))
# Verify unreviewed issues still block official export after preview.
try:
    plan(w.project)
except ValueError:
    pass
else:
    raise AssertionError('Export must require review of flagged segments')
result = w.last_result
for index, row in enumerate(w.project.rows):
    if row.issues:
        w.table.item(index, 8).setCheckState(Qt.CheckState.Checked)
assert w.last_result is result, 'Reviewing rows must not discard rendered timing'
assert w.srt_button.isEnabled()
assert len(plan(w.project)) == expected_rows
previous_project = w.project
w.translation.appendPlainText('Otra frase.')
assert w.project is previous_project, 'Editing input must preserve the previous project for saving'
assert w.save_button.isEnabled()
assert not w.export_button.isEnabled()
w.open_done(previous_project)
assert not w.srt_path.text()
w.analyze()
wait_worker()
assert not errors, errors
assert w.table.rowCount() == expected_rows, 'Reopened projects must reuse their embedded SRT'
assert w.model.currentData() == 'small'
w.dirty = False
w.close()
print('PASS: parallel-script SRT alignment, MP3 starts, preview, review state and embedded SRT reanalysis')
