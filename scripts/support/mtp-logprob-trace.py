"""Capture native target probabilities and compare the first MTP token divergence.

This tool only talks to an explicitly selected, already running local native
llama-server. It neither starts a server nor modifies model profiles. Captures
use pre-sampling logprobs (not the temperature-zero point mass), exact token IDs,
and a clean prompt cache. JSON is ASCII escaped for unambiguous token evidence.
"""
import argparse
import json
import math
from pathlib import Path
import urllib.parse
import urllib.request


def request(base_url, route, body):
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('An explicit local native server URL is required')
    if parsed.port in {11434, 11446, 11447}:
        raise ValueError('Select the isolated native runner port, never production')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base_url.rstrip('/') + route,
                                 data=json.dumps(body).encode('utf8'),
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=300) as response:
        return json.load(response)


def capture(base_url, messages, predict=128, seed=42, temperature=0, prompt_tokens=None):
    template = None
    if prompt_tokens is None:
        template = request(base_url, '/apply-template', {
            'messages': messages, 'chat_template_kwargs': {'enable_thinking': False},
            'enable_thinking': False, 'add_generation_prompt': True,
        })
        prompt_tokens = request(base_url, '/tokenize', {
            'content': template['prompt'], 'add_special': True, 'parse_special': True,
        })['tokens']
    if not prompt_tokens or any(type(x) is not int for x in prompt_tokens):
        raise ValueError('Prompt must consist of exact integer token IDs')
    body = {'prompt': prompt_tokens, 'n_predict': predict, 'temperature': temperature,
            'top_k': 20, 'top_p': .95, 'min_p': 0, 'repeat_penalty': 1,
            'presence_penalty': 0, 'frequency_penalty': 0, 'seed': seed,
            'n_probs': 8, 'post_sampling_probs': False, 'return_tokens': True,
            'cache_prompt': False, 'stream': False, 'speculative_adaptive': False}
    result = request(base_url, '/completion', body)
    tokens = result.get('tokens')
    probabilities = result.get('completion_probabilities')
    if not isinstance(tokens, list) or not isinstance(probabilities, list):
        raise ValueError('Native response is missing token or probability evidence')
    if len(tokens) != len(probabilities):
        raise ValueError('Token/probability lengths differ; no alignment may be assumed')
    for token, probability in zip(tokens, probabilities):
        if probability.get('id') != token:
            raise ValueError('Probability token IDs are misaligned')
    return {'schema_version': 1, 'prompt_tokens': prompt_tokens, 'template': template,
            'request': body, 'response': result,
            'scope': 'pre-sampling target logprobs; no assertion of quality or equivalence'}


def top_summary(probability):
    top = probability.get('top_logprobs', [])
    if len(top) < 2:
        raise ValueError('At least two pre-sampling candidates are required')
    # Some versions call this logprob, others expose raw probability as p.
    def lp(item):
        if 'logprob' in item:
            return item['logprob']
        return math.log(item.get('p', item.get('prob', 0)))
    scores = sorted([(lp(item), item) for item in top], key=lambda pair: pair[0], reverse=True)
    return {'selected_id': probability['id'],
            'selected_logprob': probability.get('logprob'),
            'top_two': [item for _, item in scores[:2]],
            'top_two_logit_gap': scores[0][0] - scores[1][0]}


def compare(baseline, candidate):
    if baseline['prompt_tokens'] != candidate['prompt_tokens']:
        raise ValueError('Prompt token IDs differ')
    a, b = baseline['response']['tokens'], candidate['response']['tokens']
    count = min(len(a), len(b))
    first = next((i for i in range(count) if a[i] != b[i]), None)
    output = {'schema_version': 1, 'same_prompt_tokens': True,
              'baseline_tokens': len(a), 'candidate_tokens': len(b),
              'identical_tokens': a == b, 'first_divergence_token': first,
              'warning': 'Top-two proximity alone cannot prove a correct recurrent state or unchanged quality'}
    if first is not None:
        pa = baseline['response']['completion_probabilities']
        pb = candidate['response']['completion_probabilities']
        output['same_forced_prefix'] = baseline['prompt_tokens'] + a[:first]
        output['preceding_token_ids'] = a[max(0, first-8):first]
        output['baseline'] = top_summary(pa[first])
        output['candidate'] = top_summary(pb[first])
        output['prefix_logprob_evidence'] = [
            {'position': i, 'baseline': top_summary(pa[i]), 'candidate': top_summary(pb[i])}
            for i in range(max(0, first-4), first+1)]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('--url', required=True)
    cap.add_argument('--messages', type=Path, required=True,
                     help='JSON messages array, or object containing messages')
    cap.add_argument('--prompt-from', type=Path)
    cap.add_argument('--predict', type=int, default=128)
    cap.add_argument('--temperature', type=float, default=0)
    cap.add_argument('--output', type=Path, required=True)
    cmp = sub.add_parser('compare')
    cmp.add_argument('--baseline', type=Path, required=True)
    cmp.add_argument('--candidate', type=Path, required=True)
    cmp.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output; evidence is never overwritten')
    if args.command == 'capture':
        messages = json.loads(args.messages.read_text(encoding='utf8'))
        if isinstance(messages, dict): messages = messages['messages']
        prompt = None
        if args.prompt_from:
            prompt = json.loads(args.prompt_from.read_text(encoding='utf8'))['prompt_tokens']
        result = capture(args.url, messages, args.predict, temperature=args.temperature,
                         prompt_tokens=prompt)
    else:
        result = compare(json.loads(args.baseline.read_text(encoding='utf8')),
                         json.loads(args.candidate.read_text(encoding='utf8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding='ascii')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {'response', 'request', 'template', 'same_forced_prefix',
                                     'prefix_logprob_evidence', 'prompt_tokens'}}, ensure_ascii=True))


if __name__ == '__main__':
    main()
