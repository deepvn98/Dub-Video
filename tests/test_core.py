import json
import math
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dub_sync.core import (
    Cancelled,
    Cue,
    DiscardedAudio,
    Project,
    Row,
    align_rows,
    align_srt_to_source_lines,
    audio_for_extra_translations,
    contiguous_srt_groups,
    extract_extra_translations,
    fingerprint,
    load_project,
    parse_srt,
    parse_parallel_lines,
    parallel_script_rows,
    plan,
    render,
    save_project,
    translated_srt,
)


class ConstantSemanticModel:
    def encode(self, texts, **kwargs):
        del kwargs
        return np.ones((len(texts), 1), dtype=float)


class CoreTests(unittest.TestCase):
    def project(self):
        return Project('', '', 6, [Cue(1, 1, 2, 'One'), Cue(2, 4, 5, 'Two')],
                       [Row('Uno', 0, 0, 0, 2, 1), Row('Dos', 1, 1, 2, 6, 1)])

    def test_invalid_srt(self):
        for raw in ['', '1\n00:00:05,000 --> 00:00:04,000\nx',
                    '1\n00:00:01,000 -->\nx',
                    '1\n00:65:00,000 --> 00:66:00,000\nx']:
            with self.assertRaises(ValueError):
                parse_srt(raw)

    def test_overlapping_srt_rejected(self):
        with self.assertRaises(ValueError):
            parse_srt('1\n00:00:01,000 --> 00:00:03,000\nx\n\n2\n00:00:02,000 --> 00:00:04,000\ny')

    def test_touching_srt_blocks_become_one_complete_group(self):
        cues = [
            Cue(13, 105.033, 107.900, 'their long tails stream behind them'),
            Cue(14, 107.900, 110.466, 'as they rise and fall through the cool'),
            Cue(15, 110.466, 112.566, 'morning air each male'),
            Cue(16, 112.566, 115.500,
                'competing for the attention of a potential mate'),
        ]
        english = [
            ('Their long tails stream behind them as they rise and fall through '
             'the cool morning air, each male competing for the attention of a potential mate.')
        ]
        spanish = [
            ('Sus largas colas se extienden detrás de ellos mientras suben y bajan '
             'por el fresco aire de la mañana, cada macho compitiendo por la '
             'atención de una posible pareja.')
        ]
        references = [
            'Sus largas colas se extienden detrás de ellos',
            'mientras suben y bajan por el fresco aire',
            'de la mañana cada macho',
            'compitiendo por la atención de una posible pareja',
        ]
        self.assertEqual(align_srt_to_source_lines(cues, english),
                         ([0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0]))
        rows = parallel_script_rows(
            cues, english, spanish, references, ConstantSemanticModel())
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].first, rows[0].last), (0, 3))
        self.assertEqual((rows[0].source_line, rows[0].source_line_last), (0, 0))
        self.assertEqual(rows[0].translation, spanish[0])

    def test_srt_grouping_uses_time_continuity(self):
        cues = [
            Cue(1, 1, 2, 'one'),
            Cue(2, 2, 3, 'two'),
            Cue(3, 3.006, 4, 'three'),
            Cue(4, 4.02, 5, 'four'),
        ]
        self.assertEqual(contiguous_srt_groups(cues), [(0, 2), (3, 3)])

    def test_non_touching_blocks_in_one_source_line_stay_separate(self):
        cues = [Cue(1, 1, 2, 'first part'), Cue(2, 3, 4, 'second part')]
        english = ['First part and second part.']
        spanish = ['Primera parte y segunda parte.']
        references = ['Primera parte', 'segunda parte']
        rows = parallel_script_rows(
            cues, english, spanish, references, ConstantSemanticModel())
        self.assertEqual([(row.first, row.last) for row in rows], [(0, 0), (1, 1)])
        self.assertEqual(' '.join(row.translation for row in rows), spanish[0])

    def test_touching_group_can_cover_multiple_parallel_lines(self):
        cues = [Cue(1, 1, 2, 'first sentence'),
                Cue(2, 2, 3, 'second sentence')]
        english = ['First sentence.', 'Second sentence.']
        spanish = ['Primera frase.', 'Segunda frase.']
        references = ['Primera frase', 'Segunda frase']
        rows = parallel_script_rows(
            cues, english, spanish, references, ConstantSemanticModel())
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].first, rows[0].last), (0, 1))
        self.assertEqual((rows[0].source_line, rows[0].source_line_last), (0, 1))
        self.assertEqual(rows[0].translation, 'Primera frase. Segunda frase.')

    def test_parallel_input_requires_one_to_one_lines(self):
        self.assertEqual(parse_parallel_lines('One.\nTwo.', 'English'),
                         ['One.', 'Two.'])
        with self.assertRaisesRegex(ValueError, 'tương ứng 1:1'):
            parallel_script_rows(
                [Cue(1, 0, 1, 'One')], ['One', 'Two'], ['Uno'], ['Uno'],
                ConstantSemanticModel())

    def test_parallel_source_line_without_srt_is_discarded(self):
        cues = [
            Cue(4, 21.2, 24.466, 'survive or fall forever'),
            Cue(5, 37.5, 40.2, 'the fallen are left behind'),
        ]
        english = [
            'Survive or fall forever.',
            'The strong push on.',
            'The fallen are left behind.',
        ]
        spanish = [
            'Sobrevivir o perecer para siempre.',
            'Los fuertes siguen adelante.',
            'Los caídos quedan atrás.',
        ]
        references = [
            'Sobrevivir o perecer para siempre',
            'Los caídos quedan atrás',
        ]
        proposed = parallel_script_rows(
            cues, english, spanish, references, ConstantSemanticModel())
        kept, discarded = extract_extra_translations(proposed)
        self.assertEqual(discarded, ['Los fuertes siguen adelante.'])
        self.assertEqual([row.translation for row in kept],
                         [spanish[0], spanish[2]])

    def test_plan_uses_target_starts(self):
        results = plan(self.project())
        self.assertEqual([r['start'] for r in results], [1, 4])
        self.assertEqual(results[0]['end'], 3)

    def test_source_starts_must_increase(self):
        p = self.project()
        p.rows[1].source_start = 0
        with self.assertRaises(ValueError):
            plan(p)

    def test_no_missing_cues(self):
        p = self.project()
        p.rows[1].first = 0
        with self.assertRaises(ValueError):
            plan(p)

    def test_review_required(self):
        p = self.project()
        p.rows[0].issues.append('Missing word')
        with self.assertRaises(ValueError):
            plan(p)
        self.assertEqual(len(plan(p, False)), 2)
        p.rows[0].reviewed = True
        self.assertEqual(len(plan(p)), 2)

    def test_missing_phrase_not_fabricated(self):
        rows = [Row('hola mundo', 0, 0), Row('completamente ausente', 1, 1)]
        segments = [{'words': [{'word': 'Hola', 'start': .2, 'end': .6}, {'word': 'mundo', 'start': .7, 'end': 1}]}]
        align_rows(rows, segments, 4)
        self.assertEqual(rows[0].confidence, 1)
        self.assertEqual(rows[1].source_end, 0)
        self.assertTrue(rows[1].issues)

    def test_repeated_phrases_remain_ordered(self):
        rows = [Row('hola mundo', 0, 0), Row('hola mundo', 1, 1)]
        segments = [{'words': [{'word': word, 'start': i, 'end': i + .6}
                               for i, word in enumerate(['hola', 'mundo', 'hola', 'mundo'])]}]
        align_rows(rows, segments, 4)
        self.assertEqual(rows[0].source_end, rows[1].source_start)
        self.assertLess(rows[1].source_end, 4)
        self.assertEqual([r.confidence for r in rows], [1, 1])

    def test_unmatched_asr_words_are_retained_by_neighboring_rows(self):
        rows = [Row('hola', 0, 0), Row('mundo', 1, 1)]
        words = ['introduccion', 'hola', 'frase', 'extra', 'mundo', 'final']
        segments = [{'words': [{'word': word, 'start': i, 'end': i + .6}
                               for i, word in enumerate(words)]}]
        align_rows(rows, segments, 7)
        self.assertEqual((rows[0].source_start, rows[0].source_end), (0, 2.8))
        self.assertAlmostEqual(rows[1].source_start, 2.8)
        self.assertAlmostEqual(rows[1].source_end, 5.72)

    def test_audio_of_rejected_translation_is_removed(self):
        rows = [Row('primera', 0, 0),
                Row('frase adicional', -1, -1, issues=['Bản dịch thừa']),
                Row('segunda', 1, 1)]
        words = ['primera', 'frase', 'adicional', 'segunda']
        segments = [{'words': [{'word': word, 'start': i, 'end': i + .6}
                               for i, word in enumerate(words)]}]
        align_rows(rows, segments, 5)
        removed = audio_for_extra_translations(rows, 5)
        kept, discarded_text = extract_extra_translations(rows)
        self.assertEqual(discarded_text, ['frase adicional'])
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(removed), 1)
        self.assertAlmostEqual(removed[0].start, .8)
        self.assertAlmostEqual(removed[0].end, 2.8)
        self.assertEqual(kept[0].source_end, removed[0].start)
        self.assertEqual(removed[0].end, kept[1].source_start)

    def test_single_missing_word_requires_review(self):
        text = 'uno dos tres cuatro cinco seis siete ocho nueve diez once doce'
        rows = [Row(text, 0, 0)]
        words = [word for word in text.split() if word != 'seis']
        segments = [{'words': [{'word': word, 'start': i, 'end': i + .6} for i, word in enumerate(words)]}]
        align_rows(rows, segments, 20)
        self.assertGreater(rows[0].confidence, .9)
        self.assertTrue(rows[0].issues)

    def test_empty_audio_alignment(self):
        with self.assertRaises(ValueError):
            align_rows([], [], 5)

    def test_invalid_loaded_project(self):
        import dataclasses
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'bad.json'
            for value in [[], {'version': 1, 'project': {}},
                          {'version': 1, 'project': {**dataclasses.asdict(self.project()), 'audio_duration': float('nan')}}]:
                file.write_text(json.dumps(value), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_project(file)

    def test_invalid_target_timing_rejected(self):
        p = self.project()
        p.cues[0].end = p.cues[0].start
        with self.assertRaises(ValueError):
            plan(p)

    def test_srt_count_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            translated_srt(self.project(), [])

    def test_invalid_audio_duration_rejected(self):
        from dub_sync.core import audio_duration
        with (patch('dub_sync.core.executable', return_value='ffprobe'),
              patch('dub_sync.core.run', return_value='{"streams":[{"codec_type":"audio"}],"format":{"duration":"NaN"}}'),
              self.assertRaises(ValueError)):
            audio_duration('audio.wav')

    def test_nonfinite_timing(self):
        p = self.project()
        p.rows[0].source_start = float('nan')
        with self.assertRaises(ValueError):
            plan(p)

    def test_roundtrip_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'project.json'
            p = self.project()
            p.model = 'medium'
            p.discarded_translations = ['Sobra.']
            p.discarded_audio = [DiscardedAudio(.1, .5, 'sobra')]
            save_project(file, p)
            self.assertEqual(load_project(file), p)

    def test_srt_uses_rendered_end(self):
        p = self.project()
        text = translated_srt(p, plan(p))
        self.assertIn('00:00:03,000', text)

    def test_cancel_preserves_existing_output(self):
        p = self.project()
        cancel = threading.Event()
        cancel.set()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.wav'
            source.write_bytes(b'source')
            p.audio = str(source)
            destination = Path(tmp) / 'result.wav'
            destination.write_bytes(b'keep')
            with self.assertRaises(Cancelled):
                render(p, destination, cancel)
            self.assertEqual(destination.read_bytes(), b'keep')

    def test_render_places_audio_and_silence(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.wav'
            import array
            samples = array.array('h', (int(10000 * math.sin(2 * math.pi * 440 * i / 44100)) for i in range(44100)))
            with wave.open(str(source), 'wb') as out:
                out.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
                out.writeframes(samples.tobytes())
            p = Project(str(source), fingerprint(source), 1, [Cue(1, 1, 3, 'hello')],
                        [Row('hola', 0, 0, 0, 1, 1)])
            result = render(p, Path(tmp) / 'result.wav')
            with wave.open(result['path']) as audio:
                self.assertEqual(audio.getnframes(), 3 * 44100)
                self.assertEqual(audio.readframes(44100), bytes(44100 * 4))
                middle = audio.readframes(44100)
                self.assertNotEqual(middle, bytes(len(middle)))
                self.assertEqual(audio.readframes(44100), bytes(44100 * 4))
            self.assertAlmostEqual(result['timings'][0]['end'], 2, places=3)

    def test_render_delays_next_clip_instead_of_cutting_or_overlapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.wav'
            import array
            samples = array.array('h', (int(6000 * math.sin(2 * math.pi * 220 * i / 44100))
                                         for i in range(2 * 44100)))
            with wave.open(str(source), 'wb') as out:
                out.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
                out.writeframes(samples.tobytes())
            cues = [Cue(1, 0, 1, 'one'), Cue(2, 1, 2, 'two')]
            rows = [Row('uno', 0, 0, 0, 1.2, 1),
                    Row('dos', 1, 1, .8, 2, 1)]
            p = Project(str(source), fingerprint(source), 2, cues, rows)
            result = render(p, Path(tmp) / 'no-overlap.wav')
            self.assertAlmostEqual(result['timings'][0]['end'], 1.2, places=3)
            self.assertAlmostEqual(result['timings'][1]['start'], 1.2, places=3)
            self.assertAlmostEqual(result['timings'][1]['delay'], .2, places=3)
            self.assertAlmostEqual(result['timings'][1]['end'], 2.4, places=3)
            with wave.open(result['path']) as audio:
                self.assertEqual(audio.getnframes(), round(2.4 * 44100))

    def test_mp3_export_uses_unmodified_audio(self):
        import array

        from dub_sync.core import audio_duration
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.wav'
            samples = array.array('h', (int(9000 * math.sin(2 * math.pi * 330 * i / 44100)) for i in range(88200)))
            with wave.open(str(source), 'wb') as out:
                out.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
                out.writeframes(samples.tobytes())
            p = Project(str(source), fingerprint(source), 2, [Cue(1, .5, 2, 'hello')],
                        [Row('hola', 0, 0, 0, 2, 1)])
            result = render(p, Path(tmp) / 'result.mp3')
            self.assertEqual(result['timings'][0]['end'], 2.5)
            self.assertLess(abs(audio_duration(result['path']) - 2.5), .1)


if __name__ == '__main__':
    unittest.main()
