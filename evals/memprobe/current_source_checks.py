"""Offline native-run checks for current/source fixes, with synthetic HTTP only."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx

from common import offline_network_guard, product_hashes
from check_memory_search import isolated_settings, seed, close
from runtime import BenchApp, OFFLINE_MOCK, answer_question, install_hooks
from timeout_policy import ObservedHTTPClient


async def run_checks(checks=None):
    checks = [] if checks is None else checks
    def check(name, value):
        checks.append({'name':'current_source_' + name, 'passed':bool(value)})
        if not value:
            raise AssertionError(name)
    install_hooks()
    offline = OFFLINE_MOCK.set(True)
    before = product_hashes()
    model = {'provider':'openai','model':'offline-fixture','base_url':'https://offline.invalid/v1',
             'api_key':'offline-fixture','api_style':'chat_completions','context_window':1000000}
    profiles = {'openai':model}
    saved = {'default_provider':'openai','providers':[model],
             'run_budget':{'mode':'custom','model_call_budget':24,'multiplier':1},
             'reflection':{'enabled':True,'inherit_main':True},'maintenance':{'enabled':True,'inherit_main':True}}
    try:
        with offline_network_guard() as network, tempfile.TemporaryDirectory(prefix='memory-current-source-') as directory:
            root = Path(directory)
            for mode in ['normal','privacy_off','rewrite_bad_json','rewrite_budget','source_tool']:
                wire = []
                def handler(request):
                    body = json.loads(request.content); wire.append(body)
                    messages = body['messages']
                    rewrite = messages[0]['content'].startswith('Rewrite only the requested property')
                    tools = [m for m in messages if m.get('role') == 'tool']
                    reads = [m for m in tools if m.get('name') == 'memory_read']
                    content, tool_calls = '{"answer":"Unknown"}', []
                    if rewrite:
                        content = 'not-json' if mode == 'rewrite_bad_json' else '{"query":"tenancy interval"}'
                    elif mode == 'source_tool' and not reads:
                        tool_calls = [{'id':'source-read','type':'function','function':{
                            'name':'memory_read','arguments':json.dumps({'memory_id':'source','purpose':'qualified'})}}]
                        content = ''
                    elif reads:
                        content = '{"answer":"monthly"}'
                    token_count = 2000000 if mode == 'rewrite_budget' and rewrite else 100
                    usage = {'prompt_tokens':token_count,'completion_tokens':10,'total_tokens':token_count+10}
                    message = {'role':'assistant','content':content}
                    if tool_calls: message['tool_calls'] = tool_calls
                    if body.get('stream'):
                        delta = {'content':content}
                        if tool_calls: delta['tool_calls'] = [{'index':i,**t} for i,t in enumerate(tool_calls)]
                        chunk={'choices':[{'index':0,'delta':delta}],'usage':usage}
                        return httpx.Response(200,text='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n',
                                              headers={'content-type':'text/event-stream'})
                    return httpx.Response(200,json={'choices':[{'message':message}],'usage':usage})
                def client(*args, **kwargs):
                    return ObservedHTTPClient(*args, **kwargs, transport=httpx.MockTransport(handler))
                app = BenchApp(isolated_settings(root/mode),saved,profiles,frozen=True,arm='native')
                try:
                    await app.initialize_benchmark()
                    if mode == 'privacy_off':
                        await app.db.set_json('privacy_settings',{'memory_in_context_enabled':False,
                            'cloud_processing_allowed':True,'auto_memory_enabled':False,'memory_review_mode':'automatic'})
                    # No initial lexical overlap: successful retrieval must use
                    # the native bounded rewrite/search/read path, not injected fixtures.
                    await seed(app,'lease',title='Occupancy agreement',summary='The tenancy interval is monthly.',
                        content='The tenancy interval is monthly.',project_id=None,source_label='',source_type='conversation',
                        fact={'subject':'user','topic':'tenancy','attribute':'tenancy interval','value':'monthly','kind':'current'},
                        fact_state='current',acceptance_source='automatic',decision_state='resolved')
                    question = 'What is my lease duration now?'
                    if mode == 'source_tool':
                        quote = 'The supplier suggested monthly billing; we have not agreed.'
                        proof = [{'id':'u','role':'user','quote':quote}]
                        target = {'subject':'user','topic':'supplies','attribute':'billing cycle','value':'monthly'}
                        await seed(app,'source',title='Supplier billing suggestion',summary=quote,content=quote,
                            provenance=proof,project_id=None,acceptance_source='automatic',decision_state='resolved',
                            fact_state='qualified',fact={**target,'kind':'reported','attribution':{
                                'origin':'third_party','speaker':'supplier','adoption':'not_stated',
                                'target':target,'provenance':proof}})
                        question = 'Who suggested my billing cycle?'
                    with patch.object(httpx,'AsyncClient',client):
                        result = await answer_question(app,question,root/mode/'answer','SYNTHETIC-CONTRACT',0,profiles)
                    check(mode+'_completed',result['status']=='completed')
                    check(mode+'_all_calls_accounted',len(result['calls'])==len(wire) and all(c.get('usage') for c in result['calls']))
                    events=result.get('events',[])
                    from verify import valid_answer_input
                    observed_calls = [json.loads(p.read_text(encoding='utf-8'))
                                      for p in (root/mode/'answer/model_calls').glob('*.json')]
                    check(mode+'_audit_accepts_native_requests', bool(observed_calls) and
                          all(valid_answer_input(c, question, result) for c in observed_calls))
                    rewrite_calls = [c for c in observed_calls if c['request']['messages'][0]['content'].startswith('Rewrite only')]
                    if rewrite_calls:
                        check(mode+'_audit_rejects_unlogged_rewrite', not valid_answer_input(rewrite_calls[0],question,{'events':[]}))
                        check(mode+'_audit_rejects_changed_question', not valid_answer_input(rewrite_calls[0],question+' changed',result))
                    rewrite_events=[e for e in events if e.get('type')=='memory_query_rewrite']
                    # Tool observations are wrapper-level; inspect the durable events as well.
                    tool_names=[(e.get('tool_call') or {}).get('name') for e in events if e.get('type')=='tool_call']
                    check(mode+'_one_search_max',tool_names.count('memory_search')<=1)
                    check(mode+'_five_reads_max',tool_names.count('memory_read')<=5)
                    total=sum(c['usage']['input_tokens'] for c in result['calls'])
                    usage_events=[e.get('usage') or {} for e in events if e.get('type') in {'assistant_message','memory_query_rewrite'}]
                    check(mode+'_run_usage_includes_rewrite',sum(u.get('input_tokens',0) for u in usage_events)==total)
                    from kore_next.run_insights import summarize_run_usage
                    summary=summarize_run_usage(events,app.run_budget)
                    check(mode+'_trace_usage_includes_rewrite',summary['main_agent']['input_tokens']==total)
                    if mode=='normal':
                        check('zero_initial_candidates',not result['candidates'])
                        check('rewrite_once',len(rewrite_events)==1)
                        check('synonym_search_reads_value','monthly' in result['answer'] and 'memory_read' in tool_names)
                    elif mode=='privacy_off':
                        check('disabled_memory_no_forced_tools',not tool_names and not rewrite_events)
                    elif mode=='rewrite_bad_json':
                        check('bad_rewrite_no_retry',len(rewrite_events)==1)
                        check('bad_rewrite_unknown_preserved','Unknown' in result['answer'])
                    elif mode=='rewrite_budget':
                        check('budget_no_post_rewrite_tools',not tool_names)
                    elif mode=='source_tool':
                        read_messages=[m for request in wire for m in request['messages'] if m.get('name')=='memory_read']
                        check('native_tool_has_claim_view',any('source_claims' in m['content'] and 'third_party' in m['content'] for m in read_messages))
                        check('source_no_query_rewrite',not rewrite_events)
                finally:
                    await close(app)
            check('network_blocked',not network)
    finally:
        OFFLINE_MOCK.reset(offline)
    check('product_sources_unchanged',product_hashes()==before)
    return checks


if __name__=='__main__':
    result=asyncio.run(run_checks())
    print(json.dumps({'status':'passed','real_api_calls':0,'checks':len(result)},ensure_ascii=False))
