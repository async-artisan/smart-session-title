"""从真实命名/归档入口验证省调用、失败记账与扫描预算，不调用外部模型。"""
import copy
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import smart-session-title as title
import archive_policy as archive
from codex_adapter import BackendError
from usage_ledger import usage_scope, usage_report, parse_usage
import test_title as fixtures
import test_archive as archive_fixtures

ID = fixtures.ID
USAGE = {'input_tokens': 37, 'cached_input_tokens': 20, 'output_tokens': 4, 'reasoning_output_tokens': 1}


class EfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.backend = fixtures.FakeBackend()
        self.config = title.DEFAULTS.copy()
        self.candidate = {'action': 'rename', 'title': '🎬 产品视频｜讲解大纲', 'reason': '明确目标'}

    def tearDown(self):
        self.tmp.cleanup()

    def process(self, model=fixtures.proposal, **kw):
        return title.process_thread(self.backend, model, ID, self.root, self.config, apply=True, **kw)

    def append(self, text='好的', **extra):
        self.backend.thread['turns'].append({'id': str(len(self.backend.thread['turns'])),
            'status': 'completed', 'items': [{'type': 'userMessage', 'content': [{'type': 'text', 'text': text}, *extra.get('parts', [])]}]})

    def actual_model(self, context):
        return title.limited_title('unused', self.root, self.config, context,
            before_model=lambda: title.ensure_title_active(self.backend, ID, self.root))

    def fake_run(self, args, **kw):
        output = Path(args[args.index('--output-last-message') + 1])
        output.write_text(json.dumps(self.candidate), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout=json.dumps({'type': 'turn.completed', 'usage': USAGE}))

    def records(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in (self.root / 'usage').glob('*/*.json')]

    def test_confirmations_skip_twice_then_model_rechecks(self):
        self.process()
        model = Mock(side_effect=fixtures.proposal)
        for text in ('好的！', 'continue.'):
            self.append(text)
            self.assertEqual(self.process(model)['skip_reason'], 'confirmation_only')
        model.assert_not_called()
        self.append('谢谢')
        self.assertNotIn('skip_reason', self.process(model))
        self.assertEqual(model.call_count, 1)
        self.assertEqual(title.read_json(title.state_path(self.root, ID))['confirmation_skips'], 0)

    def test_substantive_messages_and_attachments_still_call_model(self):
        for text, parts in [('好的，换成支付功能', []), ('推送', []), ('修复', []), ('okay?', []),
                            ('好的', [{'type': 'image', 'url': 'fixture'}]), ('改用英文命名', [])]:
            with self.subTest(text=text, parts=parts):
                self.backend = fixtures.FakeBackend()
                title.state_path(self.root, ID).unlink(missing_ok=True)
                self.process()
                self.append(text, parts=parts)
                model = Mock(side_effect=fixtures.proposal)
                self.assertNotIn('skip_reason', self.process(model))
                model.assert_called_once()

    def test_first_observation_and_policy_upgrade_require_model(self):
        self.backend.thread['name'] = self.candidate['title']
        self.backend.thread['turns'][0]['items'] = [{'type': 'userMessage', 'content': [{'type': 'text', 'text': '好的'}]}]
        model = Mock(side_effect=fixtures.proposal)
        self.process(model)
        model.assert_called_once()
        state = title.read_json(title.state_path(self.root, ID))
        state['policy_version'] = title.POLICY_VERSION - 1
        title.atomic_json(title.state_path(self.root, ID), state)
        self.append()
        self.process(model)
        self.assertEqual(model.call_count, 2)

    def test_confirmation_does_not_hide_unseen_substantive_turn(self):
        self.process()
        self.append('改为制作另一款产品的广告')
        self.append('好的')
        model = Mock(side_effect=fixtures.proposal)
        self.process(model)
        model.assert_called_once()

    def test_mutated_baseline_and_missing_baseline_cannot_skip(self):
        for change in ('content', 'id'):
            with self.subTest(change=change):
                self.backend = fixtures.FakeBackend()
                title.state_path(self.root, ID).unlink(missing_ok=True)
                self.process()
                self.append()
                if change == 'content':
                    self.backend.thread['turns'][0]['items'][0]['content'][0]['text'] = '新目标'
                else:
                    self.backend.thread['turns'][0]['id'] = 'missing'
                model = Mock(side_effect=fixtures.proposal)
                self.process(model)
                model.assert_called_once()

    def test_confirmation_still_respects_manual_title(self):
        self.process()
        self.append()
        self.backend.thread['name'] = '我的固定标题'
        model = Mock(side_effect=fixtures.proposal)
        self.assertEqual(self.process(model)['status'], 'manual_title')
        model.assert_not_called()

    def test_usage_survives_stale_result_and_keeps_no_content(self):
        def moved(args, **kwargs):
            result = self.fake_run(args, **kwargs)
            self.append('下一项工作')
            return result
        with patch('codex_adapter.subprocess.run', side_effect=moved):
            self.assertEqual(self.process(self.actual_model)['status'], 'stale_result')
        row, = self.records()
        self.assertEqual(row['usage'], USAGE)
        self.assertEqual((row['status'], row['outcome']), ('completed', 'stale_result'))
        self.assertNotIn('text', row)
        self.assertNotIn('title', row)
        self.assertNotIn('reason', row)

    def test_usage_survives_validation_error(self):
        self.candidate['title'] = '没有合法格式'
        with patch('codex_adapter.subprocess.run', side_effect=self.fake_run), self.assertRaises(ValueError):
            self.process(self.actual_model)
        row, = self.records()
        self.assertEqual(row['usage'], USAGE)
        self.assertEqual(row['outcome'], 'error')

    def test_json_failure_keeps_usage_before_parsing(self):
        def broken(args, **kwargs):
            result = self.fake_run(args, **kwargs)
            Path(args[args.index('--output-last-message') + 1]).write_text('{', encoding='utf-8')
            return result
        with patch('codex_adapter.subprocess.run', side_effect=broken), self.assertRaises(ValueError):
            self.process(self.actual_model)
        row, = self.records()
        self.assertEqual((row['status'], row['usage']), ('invalid_json', USAGE))

    def test_timeout_known_usage_and_unknown_launch_failure_are_distinct(self):
        output = json.dumps({'type': 'turn.completed', 'usage': USAGE}).encode()
        with patch('codex_adapter.subprocess.run', side_effect=subprocess.TimeoutExpired('codex', 1, output=output)):
            with self.assertRaises(BackendError): self.process(self.actual_model)
        with patch('codex_adapter.subprocess.run', side_effect=OSError('private details')):
            with self.assertRaises(OSError): self.process(self.actual_model)
        report = usage_report(self.root)
        group, = report['groups']
        self.assertEqual(group['attempts'], 2)
        self.assertEqual(group['unknown_usage'], 1)
        self.assertEqual(group['tokens'], {**USAGE, 'cache_write_input_tokens': 0})
        self.assertNotIn('private details', json.dumps(self.records()))

    def test_retry_has_two_records_without_aggregate_double_count(self):
        self.candidate = {'action': 'keep', 'title': '旧格式', 'reason': '保持'}
        with patch('codex_adapter.subprocess.run', side_effect=self.fake_run) as run:
            self.process(self.actual_model)
        self.assertEqual(run.call_count, 2)
        group, = usage_report(self.root)['groups']
        self.assertEqual(group['attempts'], 2)
        self.assertEqual(group['tokens']['input_tokens'], 74)

    def test_confirmation_and_archived_thread_create_no_model_records(self):
        with patch('codex_adapter.subprocess.run', side_effect=self.fake_run) as run:
            self.process(self.actual_model)
            self.append()
            self.process(self.actual_model)
            self.backend.archived = True
            self.process(self.actual_model)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(len(self.records()), 1)

    def test_scope_records_interruption_and_incomplete_usage(self):
        with patch('codex_adapter.subprocess.run', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): self.process(self.actual_model)
        row, = self.records()
        self.assertEqual((row['status'], row['usage']), ('interrupted', None))
        self.assertEqual(usage_report(self.root)['groups'][0]['unknown_usage'], 1)

    def test_archive_budget_prevents_reads_and_model(self):
        backend = archive_fixtures.Backend()
        with patch.dict(os.environ, {'CODEX_HOME': str(self.root / 'codex')}):
            archive.save_guards(self.root, {'pinnedThreads': [], 'threads': []})
            model = Mock()
            report = archive.scan(backend, self.root, model, budget_seconds=0)
        self.assertEqual(report['counts']['budget_exhausted'], 1)
        self.assertEqual(backend.reads, 0)
        model.assert_not_called()

    def test_archive_records_usage_even_when_content_changes_during_model(self):
        backend = archive_fixtures.Backend()
        self.candidate = {'classification': 'completed', 'reason': '已完成'}
        def moved(args, **kwargs):
            result = self.fake_run(args, **kwargs)
            backend.thread['turns'][0]['items'][0]['content'][0]['text'] = '新的未完成工作'
            return result
        with patch.dict(os.environ, {'CODEX_HOME': str(self.root / 'codex')}), patch('codex_adapter.subprocess.run', side_effect=moved):
            archive.save_guards(self.root, {'pinnedThreads': [], 'threads': []})
            result = archive.scan(backend, self.root,
                lambda context, **kw: archive.classify('unused', self.root, self.config, context, **kw))
        self.assertEqual(result['counts']['stale'], 1)
        row, = self.records()
        self.assertEqual((row['purpose'], row['outcome'], row['usage']), ('archiving', 'stale', USAGE))

    def test_archive_budget_limits_model_timeout(self):
        backend = archive_fixtures.Backend()
        self.candidate = {'classification': 'completed', 'reason': '已完成'}
        def timed(args, **kw):
            self.assertGreater(kw['timeout'], 0)
            self.assertLessEqual(kw['timeout'], 30)
            return self.fake_run(args, **kw)
        with patch.dict(os.environ, {'CODEX_HOME': str(self.root / 'codex')}), patch('codex_adapter.subprocess.run', side_effect=timed):
            archive.save_guards(self.root, {'pinnedThreads': [], 'threads': []})
            archive.scan(backend, self.root, lambda context, **kw: archive.classify('unused', self.root, self.config, context, **kw), budget_seconds=30)

    def test_usage_parser_tolerates_noise_and_rejects_invalid_fields(self):
        raw = '\n'.join(['noise', 'null', json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': -1, 'output_tokens': True}}), json.dumps({'type': 'turn.completed', 'usage': USAGE})])
        self.assertEqual(parse_usage(raw), (USAGE, False))

    def run_archive_cli(self, host, enabled=True):
        title.atomic_json(self.root / 'archive/config.json', {'enabled': enabled})
        output = io.StringIO()
        backend = archive_fixtures.Backend()
        with patch.dict(os.environ, {'CODEX_HOME': str(self.root / 'codex')}), \
             patch.object(archive, 'data_dir', return_value=self.root), \
             patch.object(archive, 'find_codex', return_value='unused') as binary, \
             patch.object(archive, 'AI助手Backend') as factory, \
             patch.object(sys, 'argv', ['archive', 'scan', '--scheduled', '--guards-stdin', '--current-id', ID, '--limit', '0']), \
             patch.object(sys, 'stdin', io.StringIO(json.dumps(host))), patch.object(sys, 'stdout', output):
            factory.return_value.__enter__.return_value = backend
            code = archive.main()
        return code, json.loads(output.getvalue()), binary

    def test_combined_guard_scan_returns_real_fresh_report(self):
        title.atomic_json(self.root / 'archive/preview.json', {'created_at': 1, 'counts': {'listed': 999}})
        code, report, _ = self.run_archive_cli({'pinnedThreads': [], 'threads': []})
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'preview')
        self.assertGreater(report['created_at'], 1)
        self.assertEqual(report['counts'], {'listed': 1, 'protected': 1})
        self.assertEqual(report, title.read_json(self.root / 'archive/preview.json'))
        self.assertIn(ID, title.read_json(self.root / 'archive/guards.json')['protected_ids'])

    def test_combined_scan_rejects_bad_or_incomplete_host_before_backend(self):
        for host in ([], {}, {'pinnedThreads': [], 'threads': [], 'unavailableHosts': ['local']}):
            with self.subTest(host=host):
                code, report, binary = self.run_archive_cli(host)
                self.assertEqual(code, 1)
                self.assertEqual(report['status'], 'error')
                binary.assert_not_called()

    def test_disabled_combined_scan_needs_no_host_or_backend(self):
        code, report, binary = self.run_archive_cli(None, enabled=False)
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'disabled')
        binary.assert_not_called()
        self.assertFalse((self.root / 'archive/guards.json').exists())


if __name__ == '__main__':
    unittest.main()
