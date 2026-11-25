"""Offline final audit; no model calls, product writes or score changes."""
from __future__ import annotations

import collections
import json
import sqlite3
from contextlib import closing

from common import HERE, RUN, dump, load_episodes, product_hashes, replay_units, sha256, utcnow
from scoring import grade
from execution import arms, validate_baseline, scoped_episodes


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    checks = []
    def check(name, ok, details=None):
        checks.append({'name': name, 'passed': bool(ok), 'details': details})

    episodes = scoped_episodes(load_episodes())
    selected = arms()
    single = selected == ['native']
    question_count = sum(len(e['probes']) for e in episodes)
    answer_count = question_count * len(selected)
    config = read(RUN / 'formal_config.json')
    progress = read(RUN / 'progress.json')
    metrics = read(RUN / 'reports/metrics.json')
    integrity = read(RUN / 'reports/integrity_audit.json')
    rows = read(RUN / 'reports/per_question.json')
    index = read(RUN / 'reports/model_call_index.json')
    expected_status = 'PILOT_COMPLETE' if config.get('execution_subset') else 'COMPLETE'
    check('complete_without_blockers', progress['status'] == expected_status and not progress['blockers'])
    check('product_sources_match_frozen_version', product_hashes() == config['product_sources'])
    check('frozen_adapter_unchanged', all(sha256(HERE / p) == h for p, h in config['adapter_sources'].items()))
    check('input_snapshot_integrity', integrity['status'] == 'passed')
    check('all_history_units_snapshots_and_questions',
          all(integrity['coverage'][k] == n for k, n in
              [('ingestion_units', sum(len(replay_units(e)[0]) for e in episodes)), ('snapshots', len(episodes)),
               ('native_questions' if single else 'paired_questions', question_count)]))
    expected = {(e['episode_id'], qi, arm) for e in episodes for qi in range(len(e['probes'])) for arm in selected}
    counts = collections.Counter((r['episode_id'], r['probe_index'], r['arm']) for r in rows)
    check('exact_unique_scored_answers', set(counts) == expected and set(counts.values()) == {1}
          and all(r['match'] != 'not_scored' for r in rows))
    call_ids = [c['id'] for c in index]
    check('unique_api_attempt_ids', len(call_ids) == len(set(call_ids)))
    check('no_inflight_model_requests', not any(c['status'] == 'started' for c in index))
    check('all_effective_answer_usage_available',
          (metrics['summary']['complete_usage_tasks'] if single else metrics['comparison']['usage_complete_pairs']) == question_count)
    if single:
        if config['evaluation'].get('baseline'):
            validate_baseline(config['evaluation'])
            comparison = metrics['historical_comparison']
            check('all_questions_compared_to_historical_baseline', comparison['paired_questions'] == question_count)
        check('progress_planned_counts', progress['answers_planned'] == answer_count and progress['questions_planned'] == question_count)
    mismatches, snapshot_errors, history_errors = [], [], []
    for episode in episodes:
        eid = episode['episode_id']
        directory = RUN / 'episodes' / eid
        with closing(sqlite3.connect((directory / 'snapshot.db').as_uri() + '?mode=ro', uri=True)) as db:
            status = db.execute('PRAGMA integrity_check').fetchone()[0]
            if status != 'ok':
                snapshot_errors.append({'episode': eid, 'status': status})
        units, _ = replay_units(episode)
        for unit in units:
            path = directory / 'ingestion/turns' / unit['unit_id'] / 'result.json'
            unit_result = read(path) if path.exists() else {}
            recorded_failure = (unit_result.get('status') == 'product_failure'
                                and unit_result.get('failure_scope') == 'native_post_run'
                                and unit_result.get('memory_processing_succeeded') is False)
            if unit_result.get('status') != 'completed' and not recorded_failure:
                history_errors.append(str(path))
        for qi, probe in enumerate(episode['probes']):
            for arm in selected:
                ad = directory / 'answers' / f'q{qi}' / arm
                result, score = read(ad / 'trial/result.json'), read(ad / 'score.json')
                answer = result.get('answer', '')
                rescored = grade(answer if result['status'] == 'completed' else '[ERROR: product execution failed]', probe, episode['latent_spec'])
                expected_score = {'episode_id': eid, 'probe_index': qi, 'arm': arm,
                                  'probe_type': probe['probe_type'], 'question': probe['question'],
                                  'gold': probe['gold_answer'], 'predicted': answer,
                                  **rescored, 'execution_status': result['status']}
                if score != expected_score:
                    mismatches.append({'episode': eid, 'question': qi, 'arm': arm})
    check('all_history_units_processed_with_failures_retained', not history_errors, history_errors)
    check('all_snapshot_databases_integrity_ok', not snapshot_errors, snapshot_errors)
    check('all_scores_reproduced_by_pinned_rule_scorer', not mismatches, mismatches)
    for field in ('input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens'):
        total = sum((c.get('usage') or {}).get(field, 0) for c in index)
        check('cost_sum_' + field, total == metrics['all_formal_usage'][field], total)
    for arm in selected:
        ar = [r for r in rows if r['arm'] == arm]
        if single:
            a = metrics['summary']
            check('native_correct_count', sum(r['correct'] for r in ar) == a['correct'])
            check('native_all_evaluated', a['evaluated'] == question_count)
            check('native_summary_by_type', all(sum(r['correct'] for r in ar if r['probe_type'] == t) == d['correct'] for t,d in a['by_type'].items()))
        else:
            a = metrics['arms'][arm]
            check(arm + '_correct_count', sum(r['correct'] for r in ar) == a['all_planned']['correct'])
            check(arm + '_effective_input_sum', sum(r['usage']['input_tokens'] for r in ar) == a['usage']['input_tokens'])
            check(arm + '_all_executed', a['executed'] == question_count and a['scored'] == question_count)
    # Validate Markdown encoding and local relative links without changing results.
    document_errors = []
    import re
    documents = [HERE.parents[1] / name for name in ('README.md', 'TEST_REPORT.md')]
    for document in documents:
        name = document.name
        text = document.read_text(encoding='utf-8', errors='strict')
        if '\ufffd' in text or re.search(r'\?{3,}', text):
            document_errors.append({'file': name, 'error': 'possible_encoding_damage'})
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if not link.startswith(('https://', 'http://', '#')) and not (document.parent / link.split('#')[0]).exists():
                document_errors.append({'file': name, 'missing_link': link})
    check('documents_utf8_and_local_links', not document_errors, document_errors)
    out = {'generated_at': utcnow(), 'status': 'passed' if all(c['passed'] for c in checks) else 'failed',
           'checks': checks, 'scorer_rechecks': answer_count, 'no_api_calls': True,
           'scope': 'Evidence, metric and package-document integrity; does not mean every answer was correct.'}
    dump(RUN / 'reports/final_audit.json', out)
    print(json.dumps({'status': out['status'], 'checks': len(checks),
                      'failures': [c for c in checks if not c['passed']]}, ensure_ascii=False))
    return 0 if out['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
