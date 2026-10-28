"""CPU-only tests for the diagnostic trace contract. No model/server access."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

SPEC=importlib.util.spec_from_file_location('mtp_trace',Path(__file__).with_name('mtp-logprob-trace.py'))
TRACE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


def probability(selected, first, second):
    return {'id':selected,'logprob':-.5,
            'top_logprobs':[{'id':first,'token':'A','logprob':-.5},
                            {'id':second,'token':'B','logprob':-.7}]}


def trace(tokens):
    return {'prompt_tokens':[1,2,3], 'response':{'tokens':tokens,
            'completion_probabilities':[probability(t,t,t+1) for t in tokens]}}


class TraceContract(unittest.TestCase):
    def test_exact_first_different_token_and_prefix(self):
        a,b=trace([7,8,9]),trace([7,10,11])
        result=TRACE.compare(a,b)
        self.assertEqual(result['first_divergence_token'],1)
        self.assertEqual(result['same_forced_prefix'],[1,2,3,7])
        self.assertEqual(result['baseline']['selected_id'],8)
        self.assertEqual(result['candidate']['selected_id'],10)
        self.assertAlmostEqual(result['baseline']['top_two_logit_gap'],.2)

    def test_mismatched_prompt_rejected(self):
        a,b=trace([7]),trace([7]); b['prompt_tokens']=[99]
        with self.assertRaisesRegex(ValueError,'Prompt token IDs differ'):
            TRACE.compare(a,b)

    def test_identical_and_shortened_responses_distinguished(self):
        a=trace([7,8])
        self.assertTrue(TRACE.compare(a,a)['identical_tokens'])
        self.assertFalse(TRACE.compare(a,trace([7]))['identical_tokens'])

    def test_capture_keeps_pre_sampling_scores_and_exact_token_ids(self):
        result=trace([7,8])['response']
        with mock.patch.object(TRACE,'request',return_value=result) as call:
            captured=TRACE.capture('http://127.0.0.1:19090',[],temperature=.6,prompt_tokens=[1,2,3])
        self.assertEqual(captured['prompt_tokens'],[1,2,3])
        body=call.call_args.args[2]
        self.assertEqual(body['prompt'],[1,2,3])
        self.assertEqual(body['temperature'],.6)
        self.assertEqual(body['top_k'],20)
        self.assertEqual(body['top_p'],.95)
        self.assertEqual(body['repeat_penalty'],1)
        self.assertFalse(body['post_sampling_probs'])
        self.assertFalse(body['cache_prompt'])

    def test_capture_rejects_misaligned_probability_evidence(self):
        result=trace([7])['response']; result['completion_probabilities'][0]['id']=8
        with mock.patch.object(TRACE,'request',return_value=result):
            with self.assertRaisesRegex(ValueError,'misaligned'):
                TRACE.capture('http://127.0.0.1:19090',[],prompt_tokens=[1,2,3])

    def test_production_and_remote_endpoints_rejected_before_network(self):
        for url in ['http://127.0.0.1:11446','http://localhost:11434',
                    'http://127.0.0.1:11447','https://127.0.0.1:19090',
                    'http://example.org:19090']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                TRACE.request(url,'/completion',{})


if __name__=='__main__':
    unittest.main()
