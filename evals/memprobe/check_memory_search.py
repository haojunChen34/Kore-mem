"""Offline end-to-end memory-tool checks; never alter frozen benchmark results."""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock

from common import HERE, RUN, dump, product_hashes, sha256, utcnow
from kore_next.agent import ToolExecutor
from kore_next.application import Application, RpcFault
from kore_next.config import Settings
from kore_next.util import dumps, now
from runtime import BenchApp, ReadOnlyMemoryTools


def isolated_settings(root):
    return Settings(data_dir=root / 'data', workspace_dir=root / 'workspace',
                    user_skills_dir=root / 'skills', api_key='', tavily_api_key='',
                    provider='openai', model='offline-fixture', base_url='http://127.0.0.1:1')


async def seed(app, memory_id, **overrides):
    stamp = now()
    item = {'id': memory_id, 'title': 'RouteFixture 项目记忆', 'summary': 'RouteFixture 使用 Zephyr。',
            'content': 'RouteFixture 使用 Zephyr 数据库，负责人为 Ada。',
            'status': 'active', 'confidence': .9, 'model_confidence': .9,
            'confidence_source': 'model_returned', 'sensitive': False,
            'conflict_status': 'none', 'conflicts': [], 'source_type': 'conversation',
            'source_id': 'synthetic-route-check', 'source_ids': ['synthetic-route-check'],
            'project_id': 'route-project', 'contacts': ['Ada'], 'evidence': ['Synthetic fixture only'],
            'created_at': stamp, 'updated_at': stamp, 'observed_at': '2026-01-15T00:00:00Z',
            'last_accessed_at': stamp, 'access_count': 0, 'revision': 1, **overrides}
    await app.db.execute('INSERT INTO memories(id,data_json,status,updated_at) VALUES(?,?,?,?)',
                         (memory_id, dumps(item), item['status'], stamp))
    await app.memory_service._index(item)
    return item


async def close(app):
    await app.mcp_manager.close()
    app.db.release_lock()


async def route_checks():
    """Exercise the native executor and the exact MemProbe subclass without APIs."""
    checks = []
    for scope in ('product', 'memprobe'):
        with tempfile.TemporaryDirectory(prefix='kore-memory-route-') as folder:
            settings = isolated_settings(Path(folder))
            app = Application(settings) if scope == 'product' else BenchApp(settings, {}, {}, frozen=True)
            try:
                if scope == 'product':
                    await app.db.initialize()
                else:
                    await app.initialize_benchmark()
                for memory_id, overrides in (
                    ('wanted', {}), ('other-project', {'project_id': 'different-project'}),
                    ('pending', {'status': 'pending_review'}), ('archived', {'status': 'archived'}),
                    ('sensitive', {'sensitive': True}),
                ):
                    await seed(app, memory_id, **overrides)
                run = await app._new_run(None, 'Offline memory route check', 'normal')
                approve = AsyncMock(return_value=False)
                executor_type = ToolExecutor if scope == 'product' else ReadOnlyMemoryTools

                def executor():
                    return executor_type(settings, approve, app.publish_artifact, app.remember, app.mcp_manager, app)

                async def case(name, action):
                    try:
                        detail = await action()
                        checks.append({'name': scope + ':' + name, 'passed': True, 'detail': detail})
                    except Exception as exc:
                        checks.append({'name': scope + ':' + name, 'passed': False,
                                       'error': f'{type(exc).__name__}: {exc}'})

                async def search(arguments, mode='normal'):
                    ex = executor()
                    catalog = json.loads(await ex.execute(run['id'], 'tool_catalog', {'query': 'memory_search'}, 'catalog', mode=mode))
                    assert 'memory_search' in catalog['activated'], catalog
                    visible = {t['function']['name'] for group in ex.tool_definitions(mode) for t in group}
                    assert 'memory_search' in visible, visible
                    raw = await ex.execute(run['id'], 'memory_search', arguments, 'search', mode=mode)
                    assert not raw.startswith('Unknown tool:'), raw
                    result = json.loads(raw)
                    assert isinstance(result.get('memories'), list), result
                    return result['memories']

                async def activation_required():
                    try:
                        await executor().execute(run['id'], 'memory_search', {'query': 'RouteFixture'}, 'inactive')
                    except PermissionError as exc:
                        assert 'not active' in str(exc), str(exc)
                        return 'Discovery remains required'
                    raise AssertionError('Unactivated tool was executed')

                async def default_search(mode='normal'):
                    rows = await search({'query': 'RouteFixture'}, mode)
                    ids = {m['id'] for m in rows}
                    assert ids == {'wanted', 'other-project'}, ids
                    assert all(m['status'] == 'active' and m['usage_state'] == 'current' for m in rows)
                    return sorted(ids)

                async def filters():
                    rows = await search({'query': 'RouteFixture', 'limit': 1,
                                         'source_type': 'conversation', 'project_id': 'route-project',
                                         'contact': 'Ada', 'since': '2026-01-01', 'until': '2026-01-31'})
                    assert [m['id'] for m in rows] == ['wanted'], rows
                    return 'All search filters forwarded'

                async def explicit_status():
                    rows = await search({'query': 'RouteFixture', 'status': 'pending_review'})
                    assert rows == [], rows
                    for purpose in ('history', 'all', 'disputes'):
                        assert await search({'query': 'RouteFixture', 'status': 'pending_review', 'purpose': purpose}) == []
                    return 'Pending memory cannot bypass approval through explicit filters'

                async def sensitive_opt_in():
                    rows = await search({'query': 'RouteFixture', 'include_sensitive': True})
                    assert {m['id'] for m in rows} == {'wanted', 'other-project', 'sensitive'}, rows
                    return 'Explicit opt-in preserved; default exclusion checked separately'

                async def empty_result():
                    assert await search({'query': 'qzxunmatchedsentinel'}) == []
                    return 'A valid query with no matches returns an empty list'

                async def invalid_arguments():
                    for arguments in ({}, {'query': ''}, {'query': None}):
                        try:
                            await search(arguments)
                        except RpcFault as exc:
                            assert 'query' in str(exc), str(exc)
                        else:
                            raise AssertionError('Missing or empty query accepted')
                    return 'Native required-argument validation reached'

                async def neighboring_tools():
                    ex = executor()
                    await ex.execute(run['id'], 'tool_catalog', {'query': 'memory'}, 'catalog')
                    listing = json.loads(await ex.execute(run['id'], 'memory_list', {'status': 'active'}, 'list'))
                    assert 'wanted' in {m['id'] for m in listing['memories']}, listing
                    detail = json.loads(await ex.execute(run['id'], 'memory_read', {'memory_id': 'wanted'}, 'read'))
                    assert detail['memory']['id'] == 'wanted', detail
                    assert 'Zephyr' in detail['memory']['content']
                    return 'Native list and read still work'

                async def write_denied():
                    ex = executor()
                    await ex.execute(run['id'], 'tool_catalog', {'query': 'memory_create'}, 'catalog')
                    before = await app.db.fetchone('SELECT COUNT(*) AS n FROM memories')
                    try:
                        await ex.execute(run['id'], 'memory_create', {'title': 'Forbidden', 'summary': 'Forbidden', 'content': 'Forbidden'}, 'write')
                    except PermissionError:
                        after = await app.db.fetchone('SELECT COUNT(*) AS n FROM memories')
                        assert before['n'] == after['n']
                        return 'Denied without writing'
                    raise AssertionError('Memory write unexpectedly permitted')

                async def plan_write_denied():
                    try:
                        await executor().execute(run['id'], 'memory_create', {}, 'plan-write', mode='plan')
                    except PermissionError:
                        return 'Plan write prohibition preserved'
                    raise AssertionError('Plan write unexpectedly permitted')

                for name, action in (
                    ('activation_required', activation_required), ('normal_search', default_search),
                    ('plan_search', lambda: default_search('plan')), ('filters_and_limit', filters),
                    ('explicit_pending_status', explicit_status), ('sensitive_opt_in', sensitive_opt_in),
                    ('empty_result', empty_result), ('required_argument', invalid_arguments),
                    ('list_and_read', neighboring_tools),
                ):
                    await case(name, action)
                checks.append({'name': scope + ':reads_need_no_approval', 'passed': approve.await_count == 0})
                await case('writes_still_denied', write_denied)
                await case('plan_writes_still_denied', plan_write_denied)
            finally:
                await close(app)
    return checks


async def replay_recorded_searches():
    """Replay recorded arguments on disposable snapshot copies, not model answers."""
    rows = json.loads((RUN / 'reports/per_question.json').read_text(encoding='utf-8'))
    attempts = []
    snapshots = []
    with tempfile.TemporaryDirectory(prefix='kore-memory-replay-') as folder:
        for directory in sorted((RUN / 'episodes').iterdir()):
            if not directory.is_dir():
                continue
            meta = json.loads((directory / 'snapshot.json').read_text(encoding='utf-8'))
            original = directory / 'snapshot.db'
            assert sha256(original) == meta['sha256'], directory.name
            episode_rows = [r for r in rows if r['episode_id'] == directory.name]
            calls = [(r, i, t) for r in episode_rows for i, t in enumerate(r['tool_calls']) if t['name'] == 'memory_search']
            if calls:
                settings = isolated_settings(Path(folder) / directory.name)
                settings.ensure_dirs()
                # sqlite3.Connection.__exit__ commits but does not close the
                # handle; close both explicitly for Windows temporary cleanup.
                with closing(sqlite3.connect(original.as_uri() + '?mode=ro', uri=True)) as source:
                    with closing(sqlite3.connect(settings.database)) as target:
                        source.backup(target)
                app = BenchApp(settings, {}, {}, frozen=True)
                try:
                    await app.initialize_benchmark()
                    run = await app._new_run(None, 'Recorded search route replay', 'normal')
                    ex = ReadOnlyMemoryTools(settings, app._request_approval, app.publish_artifact, app.remember, app.mcp_manager, app)
                    await ex.execute(run['id'], 'tool_catalog', {'query': 'memory_search'}, 'catalog')
                    for row, index, call in calls:
                        item = {'episode_id': directory.name, 'probe_index': row['probe_index'],
                                'arm': row['arm'], 'tool_index': index, 'arguments': call['arguments']}
                        try:
                            result = json.loads(await ex.execute(run['id'], 'memory_search', call['arguments'], f'replay-{index}'))
                            assert isinstance(result.get('memories'), list), result
                            item.update(passed=True, result_count=len(result['memories']),
                                        memory_ids=[m['id'] for m in result['memories']])
                        except Exception as exc:
                            item.update(passed=False, error=f'{type(exc).__name__}: {exc}')
                        attempts.append(item)
                finally:
                    await close(app)
            snapshots.append({'episode_id': directory.name, 'unchanged': sha256(original) == meta['sha256']})
    return {'attempts': attempts, 'snapshots': snapshots,
            'passed': bool(attempts) and all(x['passed'] for x in attempts) and all(x['unchanged'] for x in snapshots),
            'scope': 'Tool execution only. No LLM answers regenerated, no correctness or Token comparison inferred.'}


async def main(args):
    report = {'started_at': utcnow(), 'api_calls': 0, 'product_sources': product_hashes(),
              'checks': await route_checks(), 'frozen_run_modified': False}
    if args.replay:
        report['replay'] = await replay_recorded_searches()
    report['status'] = 'passed' if all(c['passed'] for c in report['checks']) and report.get('replay', {'passed': True})['passed'] else 'failed'
    report['finished_at'] = utcnow()
    dump(args.output, report)
    print(json.dumps({'status': report['status'], 'checks': len(report['checks']),
                      'failures': [c for c in report['checks'] if not c['passed']],
                      'replayed_searches': len(report.get('replay', {}).get('attempts', [])),
                      'output': str(args.output)}, ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--output', type=Path, default=RUN / 'tool_checks' / 'memory_search_latest.json')
    raise SystemExit(asyncio.run(main(parser.parse_args())))
