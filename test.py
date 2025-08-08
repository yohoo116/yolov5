import numpy as np


input_ref = np.load("DummyCompiler/fx_fwd_inputs.npz", allow_pickle=True)
input_ref = {key: input_ref[key] for key in input_ref.files}
output_ref = np.load("DummyCompiler/fx_fwd_outputs.npz", allow_pickle=True)
output_ref = {key: output_ref[key] for key in output_ref.files}

top_mlir_f  = "tmp_fwd_top.mlir"
tpu_mlir_f  = "tmp_fwd_bm1690_f16_tpu.mlir"
tpu_mlir_f  = "tmp_fwd_bm1690_f32_tpu.mlir"
bmodel_path = "tmp/tmp_fwd_bm1690_f32_tpu.bmodel"
graph_mod_path = "DummyCompiler/graph_module_fwd.pth"

def top_compare():
    from tpu_mlir.python.tools.model_runner import mlir_inference, model_inference
    from tpu_mlir.python.numpy_helper import npz_compare
    weights  = np.load("tmp_fwd_top_f32_all_weight.npz")
    top_outs = mlir_inference(inputs=input_ref, mlir_file=top_mlir_f, dump_all=True)
    for k in output_ref.keys():
        ref = output_ref[k]
        if k not in top_outs.keys():
            print(f"========================{k}")
            import pdb; pdb.set_trace()
            continue
        top = top_outs[k].reshape(ref.shape)
        abs_error = np.abs(top - ref)
        # 防止除以0，分母为0时相对误差设为np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(top - ref) / np.where(np.abs(ref) > 1e-12, np.abs(ref), np.nan)

        max_abs_idx = np.unravel_index(np.nanargmax(abs_error), abs_error.shape)
        max_rel_idx = np.unravel_index(np.nanargmax(rel_error), rel_error.shape)
        if np.nanmax(abs_error) > 0.1:
            print(f"{k}, 绝对误差最大值: {np.nanmax(abs_error)},ref={ref[max_abs_idx]}, top={top[max_abs_idx]}")
            print(f"     相对误差最大值: {np.nanmax(rel_error)},ref={ref[max_rel_idx]}, top={top[max_rel_idx]}")

def tpu_compare():
    from tpu_mlir.python.tools.model_runner import mlir_inference, model_inference
    from tpu_mlir.python.numpy_helper import npz_compare
    weights  = np.load("tmp_fwd_tpu_lowered_bm1690_f32_weight.npz")
    tpu_outs = mlir_inference(inputs=input_ref, mlir_file=tpu_mlir_f, dump_all=True)
    num_error = 0
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
                import pdb; pdb.set_trace()
                continue
        top = tpu_outs[tpu_k].reshape(ref.shape)
        abs_error = np.abs(top - ref)
        # 防止除以0，分母为0时相对误差设为np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(top - ref) / np.where(np.abs(ref) > 1e-12, np.abs(ref), np.nan)
        
        max_abs_idx = np.unravel_index(np.nanargmax(abs_error), abs_error.shape)
        max_rel_idx = np.unravel_index(np.nanargmax(rel_error), rel_error.shape)
        if np.nanmax(abs_error) > 0.1:
            num_error += 1
            print(f"{k}, 绝对误差最大值: {np.nanmax(abs_error)},ref={ref[max_abs_idx]}, top={top[max_abs_idx]}")
            print(f"     相对误差最大值: {np.nanmax(rel_error)},ref={ref[max_rel_idx]}, top={top[max_rel_idx]}")
    print(num_error)

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
        # 防止除以0，分母为0时相对误差设为np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(top - ref) / np.where(np.abs(ref) > 1e-12, np.abs(ref), np.nan)
        
        max_abs_idx = np.unravel_index(np.nanargmax(abs_error), abs_error.shape)
        max_rel_idx = np.unravel_index(np.nanargmax(rel_error), rel_error.shape)
        if np.nanmax(abs_error) > 0.1:
            print(f"{k}, 绝对误差最大值: {np.nanmax(abs_error)},ref={ref[max_abs_idx]}, top={top[max_abs_idx]}")
            #print(f"     相对误差最大值: {np.nanmax(rel_error)},ref={ref[max_rel_idx]}, top={top[max_rel_idx]}")

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

if __name__ == "__main__":
    # top_compare()

    # tpu_compare()
    # 
    # tpuc-opt tmp_fwd_top.mlir --processor-assign="chip=bm1690 mode=F16 num_device=1 num_core=8 addr_mode=auto high_precision=False" --processor-top-optimize --convert-top-to-tpu=" asymmetric=False doWinograd=False q_group_size=0 q_symmetric=False matmul_perchannel=False gelu_mode=normal" --canonicalize --weight-fold -o tmp_fwd_bm1690_f16_tpu.mlir

    #bmodel_compare()
    bmodel_compare_torch()
    #mlir_bmodel_vs_torch_bmodel()
