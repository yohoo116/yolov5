import numpy as np


input_ref = np.load("DummyCompiler/fx_bwd_inputs.npz", allow_pickle=True)
input_ref = {key: input_ref[key] for key in input_ref.files}
output_ref = np.load("DummyCompiler/fx_bwd_outputs.npz", allow_pickle=True)
output_ref = {key: output_ref[key] for key in output_ref.files}

top_mlir_f  = "tmp_bwd_top.mlir"
tpu_mlir_f  = "tmp_bwd_bm1690_f16_tpu.mlir"
bmodel_path = "tmp/tmp_bwd_bm1690_f32_tpu.bmodel"

def top_compare():
    from tpu_mlir.python.tools.model_runner import mlir_inference, model_inference
    from tpu_mlir.python.numpy_helper import npz_compare
    top_outs = mlir_inference(inputs=input_ref, mlir_file=top_mlir_f, dump_all=False)
    for k in output_ref.keys():
        ref = output_ref[k]
        if k not in top_outs.keys():
            print(f"========================{k}")
            continue
        top = top_outs[k].reshape(ref.shape)
        abs_error = np.abs(top - ref)
        # 防止除以0，分母为0时相对误差设为np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(top - ref) / np.where(np.abs(ref) > 1e-12, np.abs(ref), np.nan)
        
        max_abs_idx = np.unravel_index(np.nanargmax(abs_error), abs_error.shape)
        max_rel_idx = np.unravel_index(np.nanargmax(rel_error), rel_error.shape)
        if np.nanmax(abs_error) > 0.01:
            print(f"{k}, 绝对误差最大值: {np.nanmax(abs_error)},ref={ref[max_abs_idx]}, top={top[max_abs_idx]}")
            print(f"     相对误差最大值: {np.nanmax(rel_error)},ref={ref[max_rel_idx]}, top={top[max_rel_idx]}")

def tpu_compare():
    from tpu_mlir.python.tools.model_runner import mlir_inference, model_inference
    from tpu_mlir.python.numpy_helper import npz_compare
    tpu_outs = mlir_inference(inputs=input_ref, mlir_file=tpu_mlir_f, dump_all=False)
    for k in output_ref.keys():
        ref = output_ref[k]
        tpu_k = k
        if tpu_k not in tpu_outs.keys():
            if tpu_k + "_f32" in tpu_outs.keys():
                tpu_k = tpu_k + "_f32"
            elif tpu_k + "_folder" in tpu_outs.keys():
                tpu_k = tpu_k + "_folder"
            else:    
                print(f"========================{k}")
                continue
        top = tpu_outs[tpu_k].reshape(ref.shape)
        abs_error = np.abs(top - ref)
        # 防止除以0，分母为0时相对误差设为np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(top - ref) / np.where(np.abs(ref) > 1e-12, np.abs(ref), np.nan)
        
        max_abs_idx = np.unravel_index(np.nanargmax(abs_error), abs_error.shape)
        max_rel_idx = np.unravel_index(np.nanargmax(rel_error), rel_error.shape)
        if np.nanmax(abs_error) > 0.1:
            print(f"{k}, 绝对误差最大值: {np.nanmax(abs_error)},ref={ref[max_abs_idx]}, top={top[max_abs_idx]}")
            print(f"     相对误差最大值: {np.nanmax(rel_error)},ref={ref[max_rel_idx]}, top={top[max_rel_idx]}")

def bmodel_compare():
    from tpu_mlir.python.tools.model_runner import mlir_inference, model_inference
    from tpu_mlir.python.numpy_helper import npz_compare
    tpu_outs = model_inference(inputs=input_ref, model_file=bmodel_path)
    for k in output_ref.keys():
        ref = output_ref[k]
        tpu_k = k
        if tpu_k not in tpu_outs.keys():
            if tpu_k + "_f32" in tpu_outs.keys():
                tpu_k = tpu_k + "_f32"
            elif tpu_k + "_folder" in tpu_outs.keys():
                tpu_k = tpu_k + "_folder"
            else:    
                print(f"========================{k}")
                continue
        top = tpu_outs[tpu_k].reshape(ref.shape)
        abs_error = np.abs(top - ref)
        # 防止除以0，分母为0时相对误差设为np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(top - ref) / np.where(np.abs(ref) > 1e-12, np.abs(ref), np.nan)

        max_abs_idx = np.unravel_index(np.nanargmax(abs_error), abs_error.shape)
        max_rel_idx = np.unravel_index(np.nanargmax(rel_error), rel_error.shape)
        if np.nanmax(abs_error) > 0.1:
            print(f"{k}, 绝对误差最大值: {np.nanmax(abs_error)},ref={ref[max_abs_idx]}, top={top[max_abs_idx]}")
            print(f"     相对误差最大值: {np.nanmax(rel_error)},ref={ref[max_rel_idx]}, top={top[max_rel_idx]}")

def bmodel_compare_torch():
    import os
    os.environ["ModelRtRunWithTorchTpu"] = '1'
    os.environ["TorchTpuSaveKernelModule"] = '1'
    import torch
    import torch_tpu
    from torch_tpu.tpu.bmrt import BmodelModule, BmodelRuner
    device='tpu:0'
    input = torch.ones((16, 34), device=device)
    runner = BmodelRuner(bmodel_path=bmodel_path, device_id=int(device.split(':')[1]))
    ins, outs     = runner.genInplaceIO()
    inams, onames = runner.get_io_names()

    for i in range(len(ins)):
        iname = inams[i]
        ins[i].copy_(torch.from_numpy(input_ref[iname]))
    runner.forward(ins, outs)
    #outs = torch.load('t.tensor')
    for k in output_ref.keys():
        ref = output_ref[k]
        idx = None
        for i,oname in enumerate(onames):
            if k == oname or k + "_f32" == oname or k + "_folder" == oname:
                idx = i
                break
        if idx is None:
            print(f"========================{k}")
            continue
        top = outs[idx].cpu().numpy().reshape(ref.shape)
        abs_error = np.abs(top - ref)
        if np.isnan(abs_error).any() or np.isinf(abs_error).any():
            print(f"k = {k}, oname = {onames[idx]}")
            continue
        # 防止除以0，分母为0时相对误差设为np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(top - ref) / np.where(np.abs(ref) > 1e-12, np.abs(ref), np.nan)

        max_abs_idx = np.unravel_index(np.nanargmax(abs_error), abs_error.shape)
        max_rel_idx = np.unravel_index(np.nanargmax(rel_error), rel_error.shape)
        if np.isnan(top[max_abs_idx]).any() or np.isinf(top[max_abs_idx]).any():
            print(f"{k} is nan")
        if np.nanmax(rel_error) > 1:
            print(f"{k}, 绝对误差最大值: {np.nanmax(abs_error)},ref={ref[max_abs_idx]}, top={top[max_abs_idx]}")
            #print(f"{k}, 相对误差最大值: {np.nanmax(rel_error)},ref={ref[max_rel_idx]}, top={top[max_rel_idx]}")

def mlir_bmodel_vs_torch_bmodel():
    import os
    os.environ["ModelRtRunWithTorchTpu"] = '1'
    os.environ["TorchTpuSaveKernelModule"] = '1'
    device='tpu:0'

    import torch
    import torch_tpu
    from torch_tpu.tpu.bmrt import BmodelModule, BmodelRuner
    torch_tpu.tpu.set_device(device)
    # mlir outs
    mlir_outs = model_inference(inputs=input_ref, model_file=bmodel_path)

    # torch outs
    runner = BmodelRuner(bmodel_path=bmodel_path, device_id=int(device.split(':')[1]))
    ins, outs     = runner.genInplaceIO()
    inams, onames = runner.get_io_names()

    for i in range(len(ins)): ins[i].copy_(torch.from_numpy(input_ref[inams[i]]))
    runner.forward(ins, outs)
    #outs = torch.load('t.tensor')

    for k in mlir_outs.keys():
        idx = None
        for i,oname in enumerate(onames):
            if k == oname or k + "_f32" == oname or k + "_folder" == oname:
                idx = i
                break
        if idx is None:
            print(f"========================{k}")
            continue
        ref = mlir_outs[k]
        top = outs[idx].cpu().numpy().reshape(ref.shape)
        abs_error = np.abs(top - ref)

        # 防止除以0，分母为0时相对误差设为np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(top - ref) / np.where(np.abs(ref) > 1e-12, np.abs(ref), np.nan)
        
        max_abs_idx = np.unravel_index(np.nanargmax(abs_error), abs_error.shape)
        max_rel_idx = np.unravel_index(np.nanargmax(rel_error), rel_error.shape)
        if np.nanmax(abs_error) > 0.1:
            print(f"{k}, 绝对误差最大值: {np.nanmax(abs_error)},ref={ref[max_abs_idx]}, top={top[max_abs_idx]}")
            print(f"     相对误差最大值: {np.nanmax(rel_error)},ref={ref[max_rel_idx]}, top={top[max_rel_idx]}")

def test_conv():
    import torch
    import torch_tpu
    import torch.nn as nn
    import copy
    from torch_tpu.dynamo import aot_backend, dummy_backend

    device = 'tpu'
    inp = torch.randn((8,3,640,640))
    inp_tpu = copy.deepcopy(inp).to(device)
    
    inp.requires_grad = True
    inp_tpu.requires_grad = True
    
    net = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=6, padding=2,stride=2)
    net_tpu = copy.deepcopy(net).to(device)
    # net.bias.requires_grad=False
    # net_tpu.bias.requires_grad=False

    net_opt = torch.compile(net_tpu, backend=aot_backend, dynamic=None, fullgraph=False)

    out = net(inp)
    
    #out_tpu = net_tpu(inp_tpu)
    out_tpu = net_opt(inp_tpu)
    
    grad_o = torch.rand_like(out)
    grad_o_cpu = grad_o.to(device)

    out.backward(grad_o)
    out_tpu.backward(grad_o_cpu)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    
    #top_compare()
    #tpu_compare()
    
    #bmodel_compare()
    bmodel_compare_torch()
    #mlir_bmodel_vs_torch_bmodel()
    #test_conv()