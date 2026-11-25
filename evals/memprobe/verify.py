"""Offline integrity audit of dataset, native inputs, paired snapshots and tools."""
from __future__ import annotations

import json
from pathlib import Path

from common import RUN, dump, digest, load_episodes, product_hashes, replay_units, sha256, utcnow
from runtime import memory_fingerprint
from execution import arms


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def valid_answer_input(call, question, result):
    request = call.get('request', {})
    messages = request.get('messages', [])
    users = [m.get('content') for m in messages if m.get('role') == 'user']
    if users == [question]:
        return True
    # The native bounded paraphrase has its own request format. Accept only a
    # request present verbatim in the runtime ledger, retaining the exact probe.
    if (len(messages) != 2 or [m.get('role') for m in messages] != ['system', 'user']
            or not str(messages[0].get('content', '')).startswith('Rewrite only the requested property')
            or request.get('tools')):
        return False
    try:
        payload = json.loads(users[0])
    except (ValueError, TypeError, IndexError):
        return False
    if (not isinstance(payload, dict) or set(payload) != {'query', 'established_scope'}
            or payload['query'] != question or not isinstance(payload['established_scope'], str)):
        return False
    return any(e.get('type') == 'model_request' and e.get('origin') == 'memory_query_rewrite'
               and e.get('request', {}).get('messages') == messages
               and not e.get('request', {}).get('tools') for e in result.get('events', []))


def main():
    es = load_episodes()
    selected = arms()
    single = selected == ["native"]
    checks = []
    def check(name, condition, details=None):
        checks.append({"name": name, "passed": bool(condition), "details": details})
    baseline=read(RUN/'product_baseline.json')
    check('product_sources_unchanged',baseline['product_sources']==product_hashes())
    planned = sum(len(e['probes']) for e in es)
    check('full_question_manifest',len(read(RUN/'question_manifest.json'))==planned)
    allowed={'tool_catalog','memory_list','memory_search','memory_read'}
    audits={'ingestion_units':0,'answer_calls':0,'snapshots':0,'paired_questions':0,'native_questions':0,'unexpected_input':[],'forbidden_tools':[]}
    for e in es:
        eid=e['episode_id'];directory=RUN/'episodes'/eid
        units,orphans=replay_units(e)
        # Every source message appears once in the audit, including orphan and
        # tool messages that the native reflection interface does not receive.
        raw=[m for u in units for m in u['raw']]+orphans
        check(eid+'_all_raw_messages_retained',len(raw)==sum(len(s['dialogue']) for s in e['sessions']))
        for u in units:
            p=directory/'ingestion/turns'/u['unit_id']/'result.json'
            if not p.exists():continue
            r=read(p);audits['ingestion_units']+=1
            if r['input']!={'user':u['user'],'assistant':u['assistant']}:
                audits['unexpected_input'].append(str(p))
        if (directory/'snapshot.json').exists():
            meta=read(directory/'snapshot.json');audits['snapshots']+=1
            check(eid+'_snapshot_hash',sha256(directory/'snapshot.db')==meta['sha256'])
            check(eid+'_memory_fingerprint',memory_fingerprint(directory/'snapshot.db')==meta['memory_fingerprint'])
        for qi,probe in enumerate(e['probes']):
            results={}
            candidate_path = directory/'candidates'/f'q{qi}.json'
            if candidate_path.exists():
                frozen = read(candidate_path)
                check(eid+f'_q{qi}_candidate_notice_hash', digest([frozen['candidates'], frozen['notices']]) == frozen['fingerprint'])
            if single:
                check(eid+f'_q{qi}_no_paired_results', not any((directory/'answers'/f'q{qi}'/a).exists() for a in ('A','B')))
            for arm in selected:
                td=directory/'answers'/f'q{qi}'/arm/'trial'
                if (td/'result.json').exists():results[arm]=read(td/'result.json')
                for cp in (td/'model_calls').glob('*.json'):
                    c=read(cp);audits['answer_calls']+=1
                    if not valid_answer_input(c, probe['question'], results.get(arm, {})):
                        audits['unexpected_input'].append(str(cp))
                    bad=[t['function']['name'] for t in c['request']['tools'] if t['function']['name'] not in allowed]
                    if bad:audits['forbidden_tools'].append({'path':str(cp),'names':bad})
            if single and 'native' in results:
                result = results['native']
                audits['native_questions'] += 1
                check(eid+f'_q{qi}_native_candidate_origin', result.get('candidate_source') == 'product_runtime')
                check(eid+f'_q{qi}_independent_snapshot_start', (directory/'snapshot.json').exists()
                      and result.get('starting_memory_fingerprint') == read(directory/'snapshot.json')['memory_fingerprint'])
                check(eid+f'_q{qi}_runtime_candidate_audit', candidate_path.exists()
                      and frozen.get('source') == 'product_runtime'
                      and result['candidates'] == frozen['candidates'] and result['memory_notices'] == frozen['notices'])
            if not single and len(results)==2:
                audits['paired_questions']+=1
                check(eid+f'_q{qi}_same_candidates',results['A']['candidates']==results['B']['candidates'])
                check(eid+f'_q{qi}_same_status_notices',results['A']['memory_notices']==results['B']['memory_notices'])
                if candidate_path.exists():
                    check(eid+f'_q{qi}_matches_frozen_inputs', all(results[a]['candidates'] == frozen['candidates'] and results[a]['memory_notices'] == frozen['notices'] for a in ('A','B')))
    check('native_replay_fields_unchanged',not audits['unexpected_input'],audits['unexpected_input'])
    check('only_allowed_answer_tools',not audits['forbidden_tools'],audits['forbidden_tools'])
    out={'generated_at':utcnow(),'status':'passed' if all(c['passed'] for c in checks) else 'failed','checks':checks,'coverage':audits,
         'scope':'Integrity of available artifacts only; zero executed answers does not mean a completed evaluation.'}
    dump(RUN/'reports/integrity_audit.json',out)
    print(json.dumps({'status':out['status'],'coverage':audits,'checks':len(checks)},ensure_ascii=False))
    return 0 if out['status']=='passed' else 2


if __name__=='__main__':
    raise SystemExit(main())
