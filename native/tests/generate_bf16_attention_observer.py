#!/usr/bin/env python3
"""Generate a test-only observer by adding stores to the current kernel body.

Never edits production. Replay must match the uninstrumented Backend bitwise.
"""
import hashlib,sys
from pathlib import Path
source=Path(sys.argv[1]).read_text()
start=source.index('template <typename T, typename O = T>\n__global__ void attention_kernel(')
end=source.index('\n__global__ void attention_ordered_initialize_kernel',start)
body=source[start:end]
digest=hashlib.sha256(body.encode()).hexdigest()
def replace(old,new):
    global body
    if body.count(old)!=1:raise ValueError('Observer anchor changed: '+old)
    body=body.replace(old,new)
replace('void attention_kernel(', 'void attention_observer_kernel(')
replace('Meta mask_meta, Meta bias_meta) {', '''Meta mask_meta, Meta bias_meta,
    float* qk_trace, float* score_trace, float* current_trace, float* previous_trace,
    float* denominator_trace, float* maximum_trace, float* accumulation_trace,
    float* pre_round_trace) {''')
replace('score = partial * scale;', '''qk_trace[((b*q_tokens+query)*heads+head)*k_tokens+key] = partial;
                        score = partial * scale;''')
replace('score = __shfl_sync(0xffffffffU, score, 0);', '''score = __shfl_sync(0xffffffffU, score, 0);
                if(lane==0 && output_tile==0)
                    score_trace[((b*q_tokens+query)*heads+head)*k_tokens+key] = score;''')
replace('maximum = updated_maximum;', '''maximum = updated_maximum;
                    if(lane==0 && output_tile==0) {
                        const int i=((b*q_tokens+query)*heads+head)*k_tokens+key;
                        current_trace[i]=current_scale; previous_trace[i]=previous_scale;
                        denominator_trace[i]=denominator; maximum_trace[i]=maximum;
                    }''')
replace('output[oi] = store_value<O>(accumulator[item] / denominator);', '''accumulation_trace[oi]=accumulator[item];
                pre_round_trace[oi]=accumulator[item] / denominator;
                output[oi] = store_value<O>(accumulator[item] / denominator);''')
Path(sys.argv[2]).write_text('// Test observer source kernel SHA256: '+digest+'\nnamespace vrhino { namespace {\n'+body+'\n}}\n')
print(digest)
