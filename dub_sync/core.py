from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import wave
from itertools import groupby, pairwise
from pathlib import Path

# Windows can cache Hugging Face models without symlinks. Avoid presenting this
# harmless implementation detail as a warning to end users.
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled("Đã hủy tác vụ.")


@dataclasses.dataclass
class Cue:
    number: int
    start: float
    end: float
    text: str


@dataclasses.dataclass
class Row:
    translation: str
    first: int
    last: int
    source_start: float = 0
    source_end: float = 0
    confidence: float = 0
    issues: list[str] = dataclasses.field(default_factory=list)
    reviewed: bool = False
    source_line: int = -1
    source_line_last: int = -1


@dataclasses.dataclass
class DiscardedAudio:
    start: float
    end: float
    text: str


@dataclasses.dataclass
class Project:
    audio: str
    audio_hash: str
    audio_duration: float
    cues: list[Cue]
    rows: list[Row]
    model: str = "small"
    discarded_translations: list[str] = dataclasses.field(default_factory=list)
    discarded_audio: list[DiscardedAudio] = dataclasses.field(default_factory=list)
    source_script: str = ''
    spanish_script: str = ''


def timestamp(t: float) -> str:
    ms = max(0, round(t * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def read_text(path):
    raw = Path(path).read_bytes()
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        return raw.decode('utf-16')
    try:
        return raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise ValueError("Vui lòng lưu văn bản ở định dạng UTF-8 hoặc UTF-16.")


def parse_srt(text: str) -> list[Cue]:
    pattern = re.compile(r"^(\d{1,3}):(\d{2}):(\d{2})[,.](\d{3})$")

    def seconds(raw):
        match = pattern.match(raw.strip())
        if not match:
            raise ValueError(f"Mốc thời gian SRT không hợp lệ: {raw}")
        h, m, s, ms = map(int, match.groups())
        if m > 59 or s > 59:
            raise ValueError("Phút hoặc giây trong SRT không hợp lệ.")
        return h * 3600 + m * 60 + s + ms / 1000

    cues = []
    for block in re.split(r"\n\s*\n", text.strip().replace('\r', '')):
        lines = block.strip().splitlines()
        if len(lines) < 3 or not lines[0].strip().isdigit() or '-->' not in lines[1]:
            raise ValueError("SRT không hợp lệ: mỗi đoạn cần số thứ tự, thời gian và nội dung.")
        left, right = lines[1].split('-->', 1)
        end_parts = right.split()
        if not end_parts:
            raise ValueError('SRT thiếu mốc thời gian kết thúc.')
        cue = Cue(int(lines[0]), seconds(left), seconds(end_parts[0]), ' '.join(lines[2:]))
        if cue.end <= cue.start:
            raise ValueError(f"Đoạn SRT {cue.number} có thời lượng không hợp lệ.")
        if cue.number < 1:
            raise ValueError('Số thứ tự SRT phải lớn hơn 0.')
        if cues and (cue.number <= cues[-1].number or cue.start < cues[-1].end):
            raise ValueError("Bản đầu yêu cầu SRT có số tăng dần và không chồng thời gian.")
        cues.append(cue)
    if not cues:
        raise ValueError("SRT không có nội dung.")
    return cues


def word_spans(text):
    normalized = ''.join(c for c in unicodedata.normalize('NFKD', text.lower())
                         if not unicodedata.combining(c))
    return [(match.group(), match.start(), match.end())
            for match in re.finditer(r"[^\W_]+", normalized, flags=re.UNICODE)]


def extract_extra_translations(rows):
    """Remove target-only sentences while preserving them for user inspection."""
    kept = [row for row in rows if row.first >= 0]
    discarded = [row.translation for row in rows if row.first < 0]
    return kept, discarded


def audio_for_extra_translations(rows, duration):
    """Return aligned MP3 spans belonging to translation sentences rejected by SRT."""
    return [DiscardedAudio(row.source_start, row.source_end, row.translation)
            for row in rows
            if row.first < 0 and 0 <= row.source_start < row.source_end <= duration]


def load_alignment_model(cache_dir, allow_download=False):
    """Load the fixed English-to-Spanish model used only as an alignment bridge."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ValueError('Thiếu transformers. Hãy chạy setup.bat rồi mở lại Dub Sync.') from exc
    model_name = 'Helsinki-NLP/opus-mt-en-es'
    model_cache = str(Path(cache_dir) / 'alignment')
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=model_cache, local_files_only=not allow_download)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=model_cache, local_files_only=not allow_download)
        model.to('cpu')
        model.eval()
        semantic = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            cache_folder=str(Path(cache_dir) / 'semantic'), device='cpu',
            local_files_only=not allow_download)
        return tokenizer, model, semantic
    except Exception as exc:
        action = ('Bật “Cho phép tải mô hình” trong Nâng cao.' if not allow_download
                  else 'Kiểm tra kết nối mạng và dung lượng ổ đĩa.')
        raise ValueError(f'Không nạp được mô hình đối chiếu Anh–Tây Ban Nha. {action}\n{exc}') from exc


def alignment_references(cues, bundle, cancel=None,
                         progress=lambda p, s: None):
    """Translate every SRT block locally; never expose/export this text."""
    if not cues:
        raise ValueError('SRT không có block hợp lệ.')
    tokenizer, model = bundle[:2]
    source_texts = [cue.text for cue in cues]
    references = []
    batch_size = 8
    try:
        import torch
        for offset in range(0, len(source_texts), batch_size):
            check_cancel(cancel)
            batch = source_texts[offset:offset + batch_size]
            encoded = tokenizer(batch, return_tensors='pt', padding=True,
                                truncation=True, max_length=256)
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=256,
                                           num_beams=2)
            references.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            done = min(len(source_texts), offset + batch_size)
            progress(5 + int(20 * done / len(source_texts)),
                     f'Đang dịch tham chiếu block SRT {done}/{len(source_texts)}')
    except Cancelled:
        raise
    except Exception as exc:
        raise ValueError(f'Không thể tạo tham chiếu đối chiếu local:\n{exc}') from exc
    return references


def split_spanish_line_by_ranges(spanish_line, references, semantic_model,
                                 cue_ranges, cancel=None,
                                 progress=lambda p, s: None):
    """Divide one parallel Spanish line among its assigned SRT groups."""
    if len(references) != len(cue_ranges):
        raise ValueError('Số câu tham chiếu không khớp số cụm SRT.')
    document = spanish_line.strip()
    spans = word_spans(document)
    if len(spans) < len(cue_ranges):
        # A line with fewer Spanish words than SRT groups cannot provide a
        # non-empty segment for every group. Preserve the available words and
        # explicitly mark the remaining groups as missing.
        rows = []
        for index, (first, last) in enumerate(cue_ranges):
            if index < len(spans):
                char_start = spans[index][1]
                char_end = (spans[index + 1][1]
                            if index + 1 < len(spans) else len(document))
                rows.append(Row(document[char_start:char_end].strip(), first, last,
                                confidence=0,
                                issues=['Dòng Spanish quá ngắn; cần kiểm tra cách chia']))
            else:
                rows.append(Row('', first, last, issues=[
                    'Thiếu bản dịch cho cụm SRT này']))
        return rows
    target_words = [item[0] for item in spans]
    reference_words = [tokens(text) for text in references]
    ratio = len(target_words) / max(1, sum(len(value) for value in reference_words))
    flat_reference = []
    reference_owner = []
    reference_local_position = []
    for cue_index, values in enumerate(reference_words):
        flat_reference.extend(values)
        reference_owner.extend([cue_index] * len(values))
        reference_local_position.extend(range(len(values)))
    target_owner = {}
    owned_reference_positions = [set() for _ in cue_ranges]
    matcher = difflib.SequenceMatcher(None, flat_reference, target_words, autojunk=False)
    for source_start, target_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            reference_index = source_start + offset
            owner = reference_owner[reference_index]
            target_owner[target_start + offset] = owner
            owned_reference_positions[owner].add(
                reference_local_position[reference_index])
    owned_positions = [set() for _ in cue_ranges]
    for position, owner in target_owner.items():
        owned_positions[owner].add(position)

    def text_span(start, end):
        char_start = spans[start][1]
        char_end = spans[end][1] if end < len(spans) else len(document)
        return document[char_start:char_end].strip()

    reference_vectors = semantic_model.encode(
        references, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False)
    states = [(0.0, 0, [])]
    beam_width = 20
    for index, ref_words in enumerate(reference_words):
        check_cancel(cancel)
        expected = max(1, round(max(1, len(ref_words)) * ratio))
        min_length = max(1, round(expected * .45))
        max_length = max(min_length, round(expected * 1.85) + 2)
        latest_end = len(target_words) - (len(cue_ranges) - index - 1)
        candidates, metadata = [], []
        for state_index, (_, cursor, _) in enumerate(states):
            if cursor >= latest_end:
                continue
            for length in range(min_length, max_length + 1):
                end = min(latest_end, cursor + length)
                if end <= cursor:
                    continue
                candidates.append(text_span(cursor, end))
                metadata.append((state_index, cursor, end))
                if end == latest_end:
                    break
        if not candidates:
            raise ValueError('Không tìm được ranh giới bản dịch cho mọi cụm SRT.')
        vectors = semantic_model.encode(
            candidates, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False)
        next_by_cursor = {}
        for candidate_index, (state_index, start, end) in enumerate(metadata):
            old_score, _, matches = states[state_index]
            lexical = difflib.SequenceMatcher(
                None, ref_words, target_words[start:end], autojunk=False).ratio()
            semantic = float(reference_vectors[index] @ vectors[candidate_index])
            similarity = max(0.0, min(1.0, .85 * lexical + .15 * semantic))
            own = owned_positions[index]
            anchor_recall = (len(own.intersection(range(start, end))) / len(own)
                             if own else 0.0)
            contamination = sum(
                1 for position in range(start, end)
                if position in target_owner and target_owner[position] != index)
            contamination /= max(1, sum(position in target_owner
                                         for position in range(start, end)))
            length_penalty = .02 * abs(math.log(max(1, end - start) / expected))
            candidate = (old_score + similarity + 1.2 * anchor_recall
                         - 1.0 * contamination - length_penalty, end,
                         [*matches, (start, end, similarity)])
            current = next_by_cursor.get(end)
            if current is None or candidate[0] > current[0]:
                next_by_cursor[end] = candidate
        states = sorted(next_by_cursor.values(), key=lambda value: value[0], reverse=True)[:beam_width]
        progress(5 + int(20 * (index + 1) / len(cue_ranges)),
                 f'Đang tìm ranh giới cụm SRT {index + 1}/{len(cue_ranges)}')

    best = max(states, key=lambda state: state[0] - .4 *
               (len(target_words) - state[1]))
    _, _, matches = best
    matches = list(matches)

    # Repair short unanchored gaps between two well-ordered blocks. A local
    # translator may use a synonym (for example misericordia instead of
    # piedad), leaving the final target word without an exact lexical anchor.
    # Allocate those words from the unmatched tails/heads of the two reference
    # blocks, not from punctuation or user-entered line breaks.
    for index in range(len(matches) - 1):
        left_targets = owned_positions[index]
        right_targets = owned_positions[index + 1]
        left_refs = owned_reference_positions[index]
        right_refs = owned_reference_positions[index + 1]
        if not left_targets or not right_targets or not left_refs or not right_refs:
            continue
        left_anchor = max(left_targets)
        right_anchor = min(right_targets)
        gap_length = right_anchor - left_anchor - 1
        if gap_length <= 0:
            continue
        left_need = len(reference_words[index]) - max(left_refs) - 1
        right_need = min(right_refs)
        # Only repair a small wording gap. A long gap may be genuine extra
        # translation and must remain visible for review.
        if gap_length > max(2, left_need + right_need + 1):
            continue
        need = left_need + right_need
        if need:
            give_left = (gap_length * left_need + need // 2) // need
        else:
            give_left = (gap_length + 1) // 2
        boundary = left_anchor + 1 + give_left
        left_start, _, left_score = matches[index]
        _, right_end, right_score = matches[index + 1]
        if left_start < boundary < right_end:
            matches[index] = (left_start, boundary, left_score)
            matches[index + 1] = (boundary, right_end, right_score)

    # Every Spanish word in a declared 1:1 line belongs to one of that line's
    # SRT groups. Only a complete source line with no SRT is considered extra.
    if matches:
        first_start, first_end, first_score = matches[0]
        matches[0] = (0, first_end, first_score)
        last_start, _, last_score = matches[-1]
        matches[-1] = (last_start, len(target_words), last_score)

    rows = []
    for index, (start, end, similarity) in enumerate(matches):
        first, last = cue_ranges[index]
        issues = [] if similarity >= .32 else [
            f'Đối chiếu nội dung thấp {similarity:.0%}; cần kiểm tra']
        rows.append(Row(text_span(start, end), first, last,
                        confidence=similarity, issues=issues))
    return rows


def parse_parallel_lines(text, label):
    """Read one logical source/translation unit per non-empty input line."""
    result = []
    for raw in text.replace('\r', '').split('\n'):
        line = raw.strip().strip('|').strip()
        if not line or re.fullmatch(r'[-:|\s]+', line):
            continue
        if '|' in line:
            raise ValueError(f'{label} dạng bảng chỉ được có một cột.')
        result.append(line)
    if not result:
        raise ValueError(f'{label} không có nội dung.')
    return result


def contiguous_srt_groups(cues, tolerance=.01):
    """Return maximal SRT ranges whose neighboring timestamps touch."""
    if not cues:
        return []
    groups = []
    first = 0
    for index in range(1, len(cues)):
        if cues[index].start - cues[index - 1].end > tolerance:
            groups.append((first, index - 1))
            first = index
    groups.append((first, len(cues) - 1))
    return groups


def align_srt_to_source_lines(cues, source_lines):
    """Assign every SRT cue to an English source line in monotonic order."""
    if not cues or not source_lines:
        raise ValueError('Thiếu SRT hoặc kịch bản tiếng Anh gốc.')
    cue_words = [tokens(cue.text) for cue in cues]
    line_words = [tokens(line) for line in source_lines]
    if any(not words for words in cue_words + line_words):
        raise ValueError('SRT hoặc kịch bản Anh có dòng không chứa từ hợp lệ.')

    scores = []
    for cue_tokens in cue_words:
        row = []
        for source_tokens in line_words:
            matcher = difflib.SequenceMatcher(
                None, cue_tokens, source_tokens, autojunk=False)
            matched = sum(block.size for block in matcher.get_matching_blocks())
            coverage = matched / len(cue_tokens)
            row.append((coverage + .25 * matcher.ratio(), coverage))
        scores.append(row)

    states = []
    for line_index, (score, coverage) in enumerate(scores[0]):
        states.append((score - .004 * line_index,
                       [line_index], [coverage]))
    for cue_index in range(1, len(cues)):
        next_states = []
        best_prefix = None
        for line_index in range(len(source_lines)):
            candidate = states[line_index]
            if best_prefix is None or candidate[0] > best_prefix[0]:
                best_prefix = candidate
            score, coverage = scores[cue_index][line_index]
            previous_line = best_prefix[1][-1]
            skipped = max(0, line_index - previous_line - 1)
            next_states.append((
                best_prefix[0] + score - .004 * skipped,
                [*best_prefix[1], line_index],
                [*best_prefix[2], coverage],
            ))
        states = next_states
    _, assignments, confidences = max(states, key=lambda state: state[0])
    return assignments, confidences


def parallel_script_rows(cues, english_lines, spanish_lines, references,
                         semantic_model, cancel=None,
                         progress=lambda p, s: None):
    """Build one Spanish row per contiguous SRT group through parallel scripts."""
    if len(english_lines) != len(spanish_lines):
        raise ValueError(
            f'Kịch bản Anh có {len(english_lines)} dòng nhưng bản Spanish có '
            f'{len(spanish_lines)} dòng. Hai bản gốc phải tương ứng 1:1.')
    if len(references) != len(cues):
        raise ValueError('Số tham chiếu Spanish không khớp số block SRT.')
    assignments, source_confidences = align_srt_to_source_lines(
        cues, english_lines)
    groups = contiguous_srt_groups(cues)
    group_lines = []
    for first, last in groups:
        assigned = assignments[first:last + 1]
        group_lines.append((min(assigned), max(assigned)))
    by_line = [[] for _ in english_lines]
    for group_index, (line_first, line_last) in enumerate(group_lines):
        for line_index in range(line_first, line_last + 1):
            by_line[line_index].append(group_index)

    parts = []
    for line_index, group_indexes in enumerate(by_line):
        check_cancel(cancel)
        if not group_indexes:
            parts.append(Row(
                spanish_lines[line_index], -1, -1, source_line=line_index,
                source_line_last=line_index,
                issues=['Bản dịch thừa: dòng Anh gốc này không xuất hiện trong SRT']))
            continue
        line_ranges = [groups[index] for index in group_indexes]
        line_references = []
        for group_index in group_indexes:
            first, last = groups[group_index]
            relevant = [references[index] for index in range(first, last + 1)
                        if assignments[index] == line_index]
            if not relevant:
                relevant = references[first:last + 1]
            line_references.append(' '.join(relevant))
        line_rows = split_spanish_line_by_ranges(
            spanish_lines[line_index], line_references, semantic_model,
            line_ranges, cancel=cancel,
            progress=lambda _value, _message: None)
        for row in line_rows:
            row.source_line = line_index
            row.source_line_last = line_index
            if row.first >= 0:
                confidence = min(source_confidences[row.first:row.last + 1])
                if confidence < .45:
                    row.issues.append(
                        f'Khớp SRT với dòng Anh gốc thấp {confidence:.0%}; cần kiểm tra')
            parts.append(row)
        progress(26 + int(4 * (line_index + 1) / len(english_lines)),
                 f'Đang chia Spanish theo dòng Anh {line_index + 1}/{len(english_lines)}')

    result = []
    for row in parts:
        if (row.first >= 0 and result and result[-1].first == row.first
                and result[-1].last == row.last):
            previous = result[-1]
            previous.translation = f'{previous.translation} {row.translation}'.strip()
            previous.confidence = min(previous.confidence, row.confidence)
            previous.source_line_last = row.source_line_last
            previous.issues.extend(issue for issue in row.issues
                                   if issue not in previous.issues)
        else:
            result.append(row)
    return result


def fingerprint(path, cancel=None):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            check_cancel(cancel)
            digest.update(block)
    return digest.hexdigest()


def executable(name):
    found = shutil.which(name)
    if not found:
        raise ValueError(f"Không tìm thấy {name}. Cài FFmpeg và thêm thư mục bin vào PATH.")
    return found


def run(args, cancel=None):
    check_cancel(cancel)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(args, stdout=out, stderr=err, creationflags=flags)
        try:
            while True:
                try:
                    proc.wait(timeout=.15)
                    break
                except subprocess.TimeoutExpired:
                    check_cancel(cancel)
            check_cancel(cancel)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read().decode('utf-8', errors='replace'), err.read().decode('utf-8', errors='replace')
        if proc.returncode:
            raise ValueError(f"Xử lý âm thanh thất bại:\n{stderr[-2500:]}")
        return stdout


def audio_duration(path, cancel=None):
    info = json.loads(run([executable('ffprobe'), '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)], cancel))
    if not any(s.get('codec_type') == 'audio' for s in info['streams']):
        raise ValueError("File không có luồng âm thanh.")
    duration = float(info['format']['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Thời lượng âm thanh không hợp lệ.')
    return duration


def tokens(text):
    text = ''.join(c for c in unicodedata.normalize('NFKD', text.lower()) if not unicodedata.combining(c))
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def flatten_words(segments):
    words = []
    unit = 0
    for segment in segments:
        for word in segment.get('words', []):
            for token in tokens(word['word']):
                words.append({'token': token, 'start': float(word['start']),
                              'end': float(word['end']), 'unit': unit})
            unit += 1
    if not words:
        raise ValueError("Không nhận dạng được từ nào trong âm thanh.")
    if any(not math.isfinite(w['start']) or not math.isfinite(w['end']) or w['end'] < w['start'] for w in words):
        raise ValueError("Mốc từ nhận dạng không hợp lệ.")
    if words[0]['start'] < 0 or any(b['start'] < a['start'] for a, b in pairwise(words)):
        raise ValueError('Mốc từ nhận dạng không đúng thứ tự.')
    return words


def align_rows(rows, segments, duration):
    """Align translated rows to MP3 while retaining every recognized spoken word.

    The MP3 is generated from the supplied translation, so an ASR substitution or
    unmatched word is not evidence that audio is extra. Unanchored words are owned
    by the nearest matched translated row. Audio is discarded later only when its
    row belongs to translation text that was already rejected against the SRT.
    """
    if not rows:
        raise ValueError('Bản dịch không có cụm lời.')
    heard = flatten_words(segments)
    expected, ranges = [], []
    for row in rows:
        ts = tokens(row.translation)
        if not ts:
            ranges.append(None)
            continue
        ranges.append((len(expected), len(expected) + len(ts)))
        expected.extend(ts)
    if not expected:
        raise ValueError('Không có câu dịch nào để đối chiếu với MP3.')
    matcher = difflib.SequenceMatcher(None, expected, [w['token'] for w in heard], autojunk=False)
    anchors = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            anchors[block.a + offset] = block.b + offset
    expected_owner = {}
    for row_index, bounds in enumerate(ranges):
        if bounds is not None:
            expected_owner.update((position, row_index)
                                  for position in range(bounds[0], bounds[1]))
    heard_anchors = sorted(
        (heard_index, expected_owner[expected_index])
        for expected_index, heard_index in anchors.items()
        if expected_index in expected_owner)
    assigned = [[] for _ in rows]
    if heard_anchors:
        # Partition the complete recognized stream at midpoints between anchors.
        # This keeps words Whisper omitted/substituted inside a neighboring block
        # instead of silently deleting them from the rendered dub track.
        anchor_cursor = 0
        owners = []
        for heard_index in range(len(heard)):
            while (anchor_cursor + 1 < len(heard_anchors)
                   and abs(heard_anchors[anchor_cursor + 1][0] - heard_index)
                   < abs(heard_anchors[anchor_cursor][0] - heard_index)):
                anchor_cursor += 1
            owners.append(heard_anchors[anchor_cursor][1])
        # All normalized tokens originating from one Whisper word must stay in
        # the same clip; otherwise a compound/apostrophe word can be split.
        for _, run in groupby(range(len(heard)), lambda value: heard[value]['unit']):
            indexes = list(run)
            owner = max(set(owners[index] for index in indexes),
                        key=lambda value: sum(owners[index] == value for index in indexes))
            assigned[owner].extend(indexes)
    for row, bounds in zip(rows, ranges):
        structural_issues = list(row.issues)
        row.issues = structural_issues
        row.reviewed = False
        if bounds is None:
            row.source_start = row.source_end = row.confidence = 0
            continue
        a, b = bounds
        matched = [anchors[i] for i in range(a, b) if i in anchors]
        row.confidence = len(matched) / (b - a)
        if not matched:
            row.source_start = row.source_end = 0
            row.issues.append('Không tìm thấy điểm bắt đầu cụm trong MP3; cần đặt thủ công')
            continue
        lo = min(matched)
        hi = max(matched)
        row.source_start = max(0, heard[lo]['start'] - .035)
        row.source_end = min(duration, heard[hi]['end'] + .06)
        if row.confidence < 1:
            row.issues.append(f'Khớp từ {row.confidence:.0%}; kiểm tra thiếu/khác lời')
        if a not in anchors:
            row.issues.append('Chưa chắc điểm bắt đầu cụm')
    populated = [(row, indexes) for row, indexes in zip(rows, assigned)
                 if indexes and row.source_end > row.source_start]
    if populated:
        populated[0][0].source_start = max(0, heard[populated[0][1][0]]['start'] - .08)
        for (left_row, left_indexes), (right_row, right_indexes) in pairwise(populated):
            left_end = heard[left_indexes[-1]]['end']
            right_start = heard[right_indexes[0]]['start']
            # Use one shared cut in the gap between complete recognized words.
            # A shared boundary avoids both clipping and duplicated syllables.
            boundary = max(left_row.source_start + .001,
                           min(duration, (left_end + right_start) / 2))
            left_row.source_end = boundary
            right_row.source_start = boundary
        last_row, last_indexes = populated[-1]
        last_row.source_end = min(duration, heard[last_indexes[-1]]['end'] + .12)


def transcribe(path, cache_dir, model='small', language='es', allow_download=False, cancel=None, progress=lambda p, s: None):
    digest = fingerprint(path, cancel)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Text alignment is phase 1 (up to 31%). MP3 is
    # deliberately not examined until that mapping is complete.
    progress(32, f'Đang nạp mô hình {model} trên CPU; lần đầu có thể cần tải dữ liệu.')
    try:
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
        from huggingface_hub import logging as hf_logging
        hf_logging.set_verbosity_error()
    except ModuleNotFoundError as exc:
        raise ValueError('Thiếu thư viện faster-whisper. Hãy chạy setup.bat một lần rồi mở lại Dub Sync.') from exc
    try:
        model_path = None
        for root in (str(cache_dir / 'models'), None):
            try:
                model_path = download_model(model, cache_dir=root, local_files_only=True)
                break
            except Exception:
                pass
        if model_path is None:
            if not allow_download:
                raise ValueError('Mô hình chưa có trên máy.')
            model_path = download_model(model, cache_dir=str(cache_dir / 'models'), local_files_only=False)
        recognizer = WhisperModel(model_path, device='cpu', compute_type='int8')
    except Exception as exc:
        raise ValueError('Không nạp được mô hình. Nếu chưa có mô hình, bật “Cho phép tải mô hình” trong Nâng cao. '
                         'Chương trình hiện chỉ sử dụng CPU.\n' + str(exc)) from exc
    check_cancel(cancel)
    duration = audio_duration(path, cancel)

    def recognize(instance):
        segments, _ = instance.transcribe(str(path), language=language, word_timestamps=True, beam_size=5)
        result = []
        for segment in segments:
            check_cancel(cancel)
            result.append(dataclasses.asdict(segment))
            progress(min(88, 32 + int(56 * segment.end / duration)),
                     f'Nhận dạng MP3 {segment.end:.0f}/{duration:.0f} giây')
        return result

    data = recognize(recognizer)
    check_cancel(cancel)
    flatten_words(data)
    return data, digest


def validate_project(project):
    """Validate persisted project data before it is saved or rendered."""
    if not project.cues or not project.rows:
        raise ValueError('Dự án không có nội dung.')
    if not math.isfinite(project.audio_duration) or project.audio_duration <= 0:
        raise ValueError('Thời lượng âm thanh trong dự án không hợp lệ.')
    if project.model not in ('small', 'medium'):
        raise ValueError('Mô hình nhận dạng không hợp lệ.')
    if not isinstance(project.source_script, str):
        raise ValueError('Kịch bản tiếng Anh gốc trong dự án không hợp lệ.')
    if not isinstance(project.spanish_script, str):
        raise ValueError('Bản Spanish gốc trong dự án không hợp lệ.')
    if (not isinstance(project.discarded_translations, list)
            or not all(isinstance(text, str) and tokens(text) for text in project.discarded_translations)):
        raise ValueError('Danh sách câu dịch bị loại không hợp lệ.')
    if not isinstance(project.discarded_audio, list):
        raise ValueError('Danh sách đoạn MP3 bị loại không hợp lệ.')
    for item in project.discarded_audio:
        if (not isinstance(item, DiscardedAudio)
                or not all(math.isfinite(t) for t in (item.start, item.end))
                or not 0 <= item.start < item.end <= project.audio_duration + .02
                or not isinstance(item.text, str)):
            raise ValueError('Danh sách đoạn MP3 bị loại không hợp lệ.')
    previous = None
    for cue in project.cues:
        if (type(cue.number) is not int or cue.number < 1
                or not all(math.isfinite(t) for t in (cue.start, cue.end))
                or not 0 <= cue.start < cue.end or not isinstance(cue.text, str)):
            raise ValueError('Dữ liệu SRT trong dự án không hợp lệ.')
        if previous and (cue.number <= previous.number or cue.start < previous.end):
            raise ValueError('SRT trong dự án bị chồng thời gian hoặc sai thứ tự.')
        previous = cue
    for row in project.rows:
        if (type(row.first) is not int or type(row.last) is not int
                or not isinstance(row.translation, str)
                or not all(math.isfinite(t) for t in (row.source_start, row.source_end, row.confidence))
                or not 0 <= row.confidence <= 1
                or not isinstance(row.issues, list) or not all(isinstance(issue, str) for issue in row.issues)
                or type(row.reviewed) is not bool
                or type(row.source_line) is not int or row.source_line < -1
                or type(row.source_line_last) is not int
                or row.source_line_last < -1
                or (row.source_line >= 0 and row.source_line_last < row.source_line)):
            raise ValueError('Dữ liệu cụm lời trong dự án không hợp lệ.')
        extra = row.first == row.last == -1 and bool(tokens(row.translation))
        mapped = 0 <= row.first <= row.last < len(project.cues)
        if not extra and not mapped:
            raise ValueError('Phạm vi SRT của cụm lời không hợp lệ.')
        if mapped and not tokens(row.translation) and not row.issues:
            raise ValueError('Cụm SRT thiếu bản dịch nhưng không có cảnh báo.')
    expected_cue = 0
    for row in project.rows:
        if row.first < 0:
            continue
        if row.first != expected_cue:
            raise ValueError('Các cụm SRT bị thiếu, lặp hoặc sai thứ tự.')
        expected_cue = row.last + 1
    if expected_cue != len(project.cues):
        raise ValueError('Chưa đối chiếu hết các block SRT.')


def structural_issue_counts(rows):
    extras = sum(row.first < 0 for row in rows)
    missing = sum(row.first >= 0 and not tokens(row.translation) for row in rows)
    return extras, missing


def plan(project, require_review=True):
    validate_project(project)
    extras, missing = structural_issue_counts(project.rows)
    if extras or missing:
        raise ValueError(f'Không thể dựng: bản dịch thừa {extras} câu chưa được loại; SRT thiếu bản dịch {missing} cụm. '
                         'Hãy bổ sung bản dịch rồi phân tích lại.')
    results = []
    placed_until = 0.0
    for i, row in enumerate(project.rows):
        previous_start = project.rows[i - 1].source_start if i else -1
        if (row.source_start < 0 or row.source_start <= previous_start
                or row.source_start >= project.audio_duration or row.source_end <= row.source_start
                or row.source_end > project.audio_duration + .02):
            raise ValueError(f'Cụm {i + 1}: điểm bắt đầu MP3 không hợp lệ hoặc không tăng dần.')
        if require_review and row.issues and not row.reviewed:
            raise ValueError(f'Cụm {i + 1} cần nghe kiểm tra và đánh dấu “Đã kiểm tra”.')
        requested_start = project.cues[row.first].start
        length = row.source_end - row.source_start
        # Preserve complete speech without changing speed. When the requested SRT
        # slot is shorter than its audio, ripple the following clip forward rather
        # than truncating a word or mixing two voices together.
        start = max(requested_start, placed_until)
        end = start + length
        results.append({'start': start, 'end': end,
                        'requested_start': requested_start,
                        'delay': start - requested_start})
        placed_until = end
    return results


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_project(path, project):
    validate_project(project)
    if Path(path).resolve() == Path(project.audio).resolve():
        raise ValueError('Không thể ghi dự án đè lên âm thanh gốc.')
    atomic_json(path, {'version': 14, 'project': dataclasses.asdict(project)})


def load_project(path):
    try:
        data = json.loads(read_text(path))
        version = data.get('version')
        if version != 14:
            raise ValueError('Phiên bản dự án không được hỗ trợ.')
        values = data['project']
        values['cues'] = [Cue(**c) for c in values['cues']]
        values['rows'] = [Row(**row) for row in values['rows']]
        values['discarded_audio'] = [DiscardedAudio(**item) for item in values.get('discarded_audio', [])]
        p = Project(**values)
        validate_project(p)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Cấu trúc file dự án không hợp lệ.') from exc
    return p


def render_clip(audio, row, destination, cancel=None):
    # Decode from the beginning to avoid MP3 keyframe seeking errors.
    chain = f'atrim=start={row.source_start:.6f}:end={row.source_end:.6f},asetpts=PTS-STARTPTS'
    run([executable('ffmpeg'), '-nostdin', '-v', 'error', '-y', '-i', str(audio), '-map', '0:a:0',
         '-af', chain, '-ar', '44100', '-ac', '2', '-c:a', 'pcm_s16le', str(destination)], cancel)


def render(project, destination, cancel=None, progress=lambda p, s: None, require_review=True):
    check_cancel(cancel)
    destination = Path(destination).resolve()
    if destination.suffix.lower() not in ('.wav', '.mp3'):
        raise ValueError('Chỉ hỗ trợ xuất WAV hoặc MP3.')
    if destination == Path(project.audio).resolve():
        raise ValueError('Không thể ghi đè âm thanh gốc.')
    if not project.audio or not project.audio_hash:
        raise ValueError('Dự án chưa có MP3 lồng tiếng đã được phân tích.')
    timings = plan(project, require_review)
    if fingerprint(project.audio, cancel) != project.audio_hash:
        raise ValueError('File âm thanh đã thay đổi. Hãy phân tích lại.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dub-render-', dir=destination.parent) as tmp:
        tmp = Path(tmp)
        wav_path = tmp / 'timeline.wav'
        raw_path = tmp / 'timeline.pcm'
        rate = 44100
        actual = []
        max_frame = round(project.cues[-1].end * rate)
        import numpy as np
        with open(raw_path, 'w+b') as mixed:
            for i, (row, timing) in enumerate(zip(project.rows, timings)):
                check_cancel(cancel)
                progress(int(90 * i / len(project.rows)), f'Đang dựng cụm {i + 1}/{len(project.rows)}')
                clip = tmp / 'clip.wav'
                render_clip(project.audio, row, clip, cancel)
                target = round(timing['start'] * rate)
                with wave.open(str(clip), 'rb') as source:
                    frames = source.getnframes()
                    remaining = frames
                    byte_offset = target * 4
                    while remaining:
                        check_cancel(cancel)
                        chunk = source.readframes(min(rate, remaining))
                        if not chunk:
                            raise ValueError(f'Cụm {i + 1}: file âm thanh tạm bị thiếu dữ liệu.')
                        mixed.seek(byte_offset)
                        existing = mixed.read(len(chunk))
                        if len(existing) < len(chunk):
                            existing += bytes(len(chunk) - len(existing))
                        source_samples = np.frombuffer(chunk, dtype='<i2').astype(np.int32)
                        old_samples = np.frombuffer(existing, dtype='<i2').astype(np.int32)
                        combined = np.clip(source_samples + old_samples, -32768, 32767).astype('<i2')
                        mixed.seek(byte_offset)
                        mixed.write(combined.tobytes())
                        byte_offset += len(chunk)
                        remaining -= len(chunk) // 4
                end_frame = target + frames
                max_frame = max(max_frame, end_frame)
                actual.append({**timing, 'end': end_frame / rate})
            mixed.seek(max_frame * 4 - 1)
            mixed.write(b'\0')
        with wave.open(str(wav_path), 'wb') as output, open(raw_path, 'rb') as mixed:
            output.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
            for chunk in iter(lambda: mixed.read(rate * 4), b''):
                check_cancel(cancel)
                output.writeframesraw(chunk)
        check_cancel(cancel)
        if destination.suffix.lower() == '.mp3':
            final = tmp / 'final.mp3'
            progress(94, 'Đang mã hóa MP3…')
            run([executable('ffmpeg'), '-nostdin', '-v', 'error', '-y', '-i', str(wav_path), '-c:a', 'libmp3lame', '-b:a', '192k', str(final)], cancel)
        else:
            final = wav_path
        check_cancel(cancel)
        os.replace(final, destination)
    progress(100, 'Đã xuất âm thanh.')
    return {'path': str(destination), 'timings': actual}


def translated_srt(project, timings):
    if len(project.rows) != len(timings):
        raise ValueError('Số cụm phụ đề không khớp kết quả dựng. Hãy dựng lại âm thanh.')
    return '\n\n'.join(f'{i}\n{timestamp(t["start"])} --> {timestamp(t["end"])}\n{row.translation}'
                        for i, (row, t) in enumerate(zip(project.rows, timings), 1)) + '\n'
