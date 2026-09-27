"""GPU regression: changed inputs and prefix lengths replay the same graph."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['JOYOMNI_SAGE_ATTN'] = '1'
import json
from pathlib import Path
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from xvideo.models.dit.prefix_attention import prepare_prefix_mask, set_prefix_length
from xvideo.models.dit.prompt_mask import masked_attention
from xvideo.models.dit.dit import Transformer3DModel
from xvideo.serving.graph_runner import StreamingGraphRunner

torch.set_num_threads(1);torch.manual_seed(123)
device=torch.device('cuda:0');dtype=torch.bfloat16
reports=[]
with torch.no_grad():
    q=torch.randn(1,129,2,128,device=device,dtype=dtype)
    k=torch.randn(1,513,2,128,device=device,dtype=dtype);v=torch.randn_like(k)
    mask=torch.ones(1,1,1,513,device=device,dtype=torch.bool)
    prepare_prefix_mask(mask,129)
    def run():return masked_attention(q,k,v,mask)
    for _ in range(3):run()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=run()
    for length in (513,65,1,511,129):
        q.normal_();k.normal_();v.normal_()
        k[:,length:]=10000;v[:,length:]=-10000
        mask.fill_(False);mask[...,:length]=True;set_prefix_length(mask,length)
        g.replay();torch.cuda.synchronize()
        ref=F.scaled_dot_product_attention(q.transpose(1,2),k[:,:length].transpose(1,2),v[:,:length].transpose(1,2)).transpose(1,2)
        relative=float((out-ref).float().norm()/ref.float().norm())
        assert torch.isfinite(out).all() and relative<.03,(length,relative)
        reports.append({'validKeys':length,'relativeL2':relative})
    model=Transformer3DModel(patch_size=[1,1,1],in_channels=4,out_channels=4,hidden_size=128,heads_num=1,text_states_dim=64,mlp_width_ratio=2,mm_double_blocks_depth=2).to(device=device,dtype=dtype).eval()
    shape=(1,4,1,2,3);capacity=512
    image=torch.randn(shape,device=device,dtype=dtype);ref_image=torch.randn_like(image)
    embeds=torch.randn(1,capacity,64,device=device,dtype=dtype)
    tm=torch.ones(1,1,1,12+capacity,device=device,dtype=torch.bool)
    prepare_prefix_mask(tm,12+capacity)
    ts=torch.ones(1,device=device,dtype=dtype)
    def full():return model(image,ts,embeds,ref_video_latent=ref_image,attention_mask=tm)[0]
    for _ in range(3):full()
    torch.cuda.synchronize();tg=torch.cuda.CUDAGraph()
    with torch.cuda.graph(tg):actual=full()
    transformer=[]
    for n in (3,512,1,475,64):
        image.normal_();ref_image.normal_()
        source=torch.randn(1,n,64,device=device,dtype=dtype)
        embeds.fill_(1000);embeds[:,:n]=source
        tm.fill_(False);tm[...,:12+n]=True;set_prefix_length(tm,12+n)
        expected=model(image,ts,source,ref_video_latent=ref_image)[0]
        tg.replay();torch.cuda.synchronize()
        relative=float((actual-expected).float().norm()/expected.float().norm())
        assert torch.isfinite(actual).all() and relative<.03,(n,relative)
        transformer.append({'tokens':n,'relativeL2':relative})
    ids=model._get_token_frame_ids((8,2,3),device,temporal_ids=torch.arange(8,device=device))
    cos,sin=model.get_rotary_pos_embed_from_ids(frame_ids=ids,spatial_shape=(2,3))
    runner=StreamingGraphRunner(model,chunk_tokens=6,ref_tokens=0,latent_shape=shape,txt_len=capacity,max_temporal_ids=8,pos_freqs=(cos.unsqueeze(0),sin.unsqueeze(0)),device=device,dtype=dtype,autocast_ctx=nullcontext,mask_prompt=True)
    runner.capture(timesteps=torch.tensor([1.],device=device),sigmas=torch.tensor([1.,0.],device=device))
    graph=runner.full_graph;address=runner.prompt_key_mask._sage_prefix[1].data_ptr()
    for n in (3,512,1,475):
        runner.bind_session(10,torch.randn(1,n,64,device=device,dtype=dtype))
        runner.in_noise.copy_(image);runner.in_ref_latent.copy_(ref_image)
        graph.replay();torch.cuda.synchronize()
        assert runner.full_graph is graph and runner.prompt_key_mask._sage_prefix[1].data_ptr()==address
        assert runner.prompt_key_mask._sage_prefix[1][1].item()==runner.prompt_key_prefix+n
        assert torch.isfinite(runner.out_latents).all()
    # Every size is captured before use. Private graph pools plus shared KV
    # must remain correct when buckets are replayed in arbitrary order.
    def make_runner(cap,shared=None):
        r=StreamingGraphRunner(model,chunk_tokens=6,ref_tokens=0,latent_shape=shape,txt_len=cap,max_temporal_ids=8,pos_freqs=(cos.unsqueeze(0),sin.unsqueeze(0)),device=device,dtype=dtype,autocast_ctx=nullcontext,mask_prompt=True,shared_kv=shared)
        r.capture(timesteps=torch.tensor([1.],device=device),sigmas=torch.tensor([1.,0.],device=device))
        return r
    family={512:runner}
    for cap in (640,768,896,1024):family[cap]=make_runner(cap,runner)
    reference=make_runner(1024)
    assert len({r.pool_k.data_ptr() for r in family.values()})==1
    assert len({r.full_graph.pool() for r in family.values()})==5
    switches=[];published={}
    for i,n in enumerate((475,1024,1,700,513,900,477,640,641,896,897,512,1024,475)):
        cap=min(c for c in family if c>=n);active=family[cap]
        embeds=torch.randn(1,n,64,device=device,dtype=dtype)
        image.normal_();ref_image.normal_()
        for r in (reference,active):
            r.bind_session(i,embeds);r.pool_k.zero_();r.pool_v.zero_()
            r.in_noise.copy_(image);r.in_ref_latent.copy_(ref_image)
            r.full_graph.replay()
        torch.cuda.synchronize()
        relative=float((active.out_latents-reference.out_latents).float().norm()/reference.out_latents.float().norm())
        assert relative<.03,(n,relative)
        captured_k=active.stage_k.clone();captured_v=active.stage_v.clone()
        eager_k=torch.empty_like(captured_k);eager_v=torch.empty_like(captured_v)
        def store(li,k,v):eager_k[li].copy_(k);eager_v[li].copy_(v)
        model._graph_kv_writer=store
        try:
            with model.cache_context('cond'):
                model(hidden_states=active.in_store_latent,timestep=active.in_store_timestep,
                      encoder_hidden_states=active.in_prompt,current_temporal_ids=active.in_current_ids,
                      cached_temporal_ids=None,kv_cache_mode='store',kv_cache_scope='cond',
                      kv_cache_chunk_id=None,kv_cache_selected_chunk_ids=[],kv_cache_pre_rope=True,skip_text_stream=True)
        finally:model._graph_kv_writer=None
        torch.testing.assert_close(captured_k,eager_k,rtol=.03,atol=.03)
        torch.testing.assert_close(captured_v,eager_v,rtol=.03,atol=.03)
        if i:
            previous=published[i-1][0]['key'].clone()
        active.publish_to_cache(published,i)
        if i:torch.testing.assert_close(previous,published[i-1][0]['key'])
        switches.append({'tokens':n,'capacity':cap,'relativeL2':relative})
fp8=os.environ.get('JOYOMNI_FP8_IMG')=='1' and os.environ.get('JOYOMNI_FP8_TXT')=='1'
report={'attention':reports,'transformer':transformer,'streamingRunnerReused':True,'fp8':fp8,'familySwitches':switches,'independentGraphPools':True,'sharedKvAndGlobalPublicationParity':True}
Path('tulip/prefix-attention'+('-fp8' if fp8 else '')+'-proof-20260918.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
