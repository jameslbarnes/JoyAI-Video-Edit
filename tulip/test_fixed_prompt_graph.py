"""CPU regressions: fixed addresses, correct masks, graph-key reuse, no truncation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from xvideo.serving.graph_runner import StreamingGraphRunner, prompt_graph_capacity, prompt_graph_capacities
from xvideo.models.dit.prompt_mask import masked_attention

class FixedPromptGraph(unittest.TestCase):
    def runner(self):
        r=StreamingGraphRunner.__new__(StreamingGraphRunner)
        r.txt_len=16;r.mask_prompt=True;r.prompt_key_prefix=7;r.dtype=torch.float64
        r.in_prompt=torch.zeros(1,16,8,dtype=r.dtype)
        r.prompt_key_mask=torch.zeros(1,1,1,23,dtype=torch.bool)
        return r

    def test_binding_shorter_and_longer_prompts_keeps_storage_and_clears_tail(self):
        r=self.runner();addresses=(r.in_prompt.data_ptr(),r.prompt_key_mask.data_ptr())
        for n in (3,16,1,11):
            x=torch.randn(1,n,8,dtype=r.dtype);r.bind_session(5,x)
            self.assertEqual(addresses,(r.in_prompt.data_ptr(),r.prompt_key_mask.data_ptr()))
            self.assertTrue(torch.equal(r.in_prompt[:,:n],x))
            self.assertEqual(torch.count_nonzero(r.in_prompt[:,n:]).item(),0)
            self.assertEqual(r.prompt_key_mask.sum().item(),7+n)
            self.assertTrue(r.prompt_key_mask[...,:7+n].all())
            self.assertEqual(r.prompt_tokens,n)

    def test_overflow_rejected_before_existing_prompt_is_modified(self):
        r=self.runner();r.bind_session(1,torch.ones(1,5,8,dtype=r.dtype));old=r.in_prompt.clone()
        with self.assertRaises(ValueError):r.bind_session(2,torch.ones(1,17,8,dtype=r.dtype))
        self.assertTrue(torch.equal(r.in_prompt,old));self.assertEqual(r.bound_token,1)

    def test_mask_removes_padding_including_large_nonzero_values(self):
        torch.manual_seed(123)
        q=torch.randn(1,15,2,8,dtype=torch.float64)
        k=torch.randn(1,23,2,8,dtype=torch.float64);v=torch.randn_like(k)
        for n in (1,6,16):
            mask=(torch.arange(23)<7+n)[None,None,None,:]
            expected=masked_attention(q,k[:,:7+n],v[:,:7+n],torch.ones(1,1,1,7+n,dtype=torch.bool))
            poison_k=k.clone();poison_v=v.clone();poison_k[:,7+n:]=10000;poison_v[:,7+n:]=-10000
            actual=masked_attention(q,poison_k,poison_v,mask)
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)

    def test_graph_key_reused_across_lengths_and_capacity_is_enforced(self):
        p=Path(__file__).resolve().parents[1]/'deploy/xvideo/serving/joyomni_streaming.py'
        tree=ast.parse(p.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JoyOmniV2VStreamingSession')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_graph_key')
        ns={'fixed_prompt_graph_enabled':lambda:True,'prompt_graph_capacity':prompt_graph_capacity}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(p),'exec'),ns)
        c=SimpleNamespace(settings=SimpleNamespace(max_sequence_length=None,max_temporal_ids=8,num_inference_steps=2),cfg=SimpleNamespace(text_token_max_length=1024),latent_h=15,latent_w=27,_ref_kv_tokens=lambda:0)
        keys=[ns['_graph_key'](c,torch.zeros(1,n,4)) for n in (1,363,444,512)]
        self.assertTrue(all(k==keys[0] for k in keys));self.assertEqual(keys[0][2],512)
        for n,capacity in ((513,640),(700,768),(1024,1024)):
            self.assertEqual(ns['_graph_key'](c,torch.zeros(1,n,4))[2],capacity)
        self.assertEqual(prompt_graph_capacity(100,200),200)
        with self.assertRaises(ValueError):ns['_graph_key'](c,torch.zeros(1,1025,4))
        prepare=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_prompt_graph')
        ns.update(torch=torch,graph_env_enabled=lambda:True)
        exec(compile(ast.Module(body=[prepare],type_ignores=[]),str(p),'exec'),ns)
        capacities=prompt_graph_capacities(1024)
        self.assertEqual(capacities,(512,640,768,896,1024))
        c.runtime=SimpleNamespace(graph_runners={(*keys[0][:2],cap,*keys[0][3:]):object() for cap in capacities})
        c._graph_key=lambda embeds:ns['_graph_key'](c,embeds)
        # Deliberately omit every drain/capture lock: changed token counts
        # must reuse the graph without attempting a live capture.
        for n in range(1,1025):ns['_prepare_prompt_graph'](c,torch.zeros(1,n,4),1)
        c.runtime.graph_runners.clear()
        with self.assertRaisesRegex(RuntimeError,'not prewarmed'):
            ns['_prepare_prompt_graph'](c,torch.zeros(1,600,4),1)


if __name__=='__main__':unittest.main()
