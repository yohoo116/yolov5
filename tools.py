import numpy as np

def compare_weight(tpu_ = 'tpu_weight.npz', cpu_ = 'cpu_weight.npz'):
    print('=============================================weight compare')
    t_w = np.load(tpu_, allow_pickle=True)
    c_w = np.load(cpu_, allow_pickle=True)

    for k in c_w.keys():
        c_g = c_w[k]
        t_g = t_w[k]
        diff = abs(c_g - t_g)
        index_abs = diff.argmax()
        #related_diff = abs(diff/c_g)
        #index_related = related_diff.argmax()
        print(k, 
                ",max abs diff: ", np.max(diff), " exp:", c_g.flatten()[index_abs], ", got:", t_g.flatten()[index_abs],
                #",max rel diff: ", np.max(related_diff), ", exp: ", c_g.flatten()[index_related], ", got:", t_g.flatten()[index_related]
            )
    print('=============================================weight compare done')

def get_model_grad(model, save_path="model_grad.npz", save_ = True):
    grad_dict = {}  
    for name, param in model.named_parameters():
        #print(name)
        if isinstance(param.grad, torch.Tensor):
            grad_dict[name] = param.grad.cpu().numpy()
        else:
            print(name, "has no grad")
    if save_:
        np.savez(save_path, **grad_dict)
    return grad_dict

def compare_grad(tpu_ = 'tpu_model_grad.npz', cpu_ = 'cpu_model_grad.npz'):
    import sys
    with open('output.txt', 'w', encoding='utf-8') as f:
        old_stdout = sys.stdout
        sys.stdout = f
        
        t_w = np.load(tpu_, allow_pickle=True)
        c_w = np.load(cpu_, allow_pickle=True)
        for k in c_w.keys():
            c_g = c_w[k]
            t_g = t_w[k]
            diff = abs(c_g - t_g)
            index_abs = diff.argmax()
            # related_diff = abs(diff/c_g)
            # index_related = related_diff.argmax()
            print(k, 
                    ",max abs diff: ", np.max(diff), " exp:", c_g.flatten()[index_abs], ", got:", t_g.flatten()[index_abs],
                    #",max rel diff: ", np.max(related_diff), ", exp: ", c_g.flatten()[index_related], ", got:", t_g.flatten()[index_related]
                )
        print('=============================================grad compare done')
        
        sys.stdout = old_stdout


def test_tpu():
    i0_weight  = np.load("model_weight_before_update_iter0.npz", allow_pickle=True)
    i0_w_names = [ k for k in i0_weight.keys()]
    i0_grad    = np.load("model_grad_iter0.npz", allow_pickle=True)
    i0_g_names = [ k for k in i0_grad.keys()]
    i1_weight  = np.load("model_weight_after_update_iter0.npz", allow_pickle=True)
    i1_w_names = [ k for k in i1_weight.keys()]

    i2_weight  = np.load("model_weight_before_update_iter1.npz", allow_pickle=True)
    i2_w_names = [ k for k in i2_weight.keys()]

    for i in range(len(i0_w_names)):
        name = i0_w_names[i]
        i0_w = i0_weight[name]
        i0_g = i0_grad[name]
        i1_w = i1_weight[name]
        i2_w = i2_weight[name]
        diff = abs(i1_w - i0_w)
        print(name, np.max(diff))

def test_cpu():
    i0_weight  = np.load("cpu_weight_before_update_iter0.npz", allow_pickle=True)
    i0_w_names = [ k for k in i0_weight.keys()]
    i0_grad    = np.load("cpu_grad_iter0.npz", allow_pickle=True)
    i0_g_names = [ k for k in i0_grad.keys()]
    i1_weight  = np.load("cpu_weight_after_update_iter0.npz", allow_pickle=True)
    i1_w_names = [ k for k in i1_weight.keys()]

    i2_weight  = np.load("cpu_weight_before_update_iter1.npz", allow_pickle=True)
    i2_w_names = [ k for k in i2_weight.keys()]

    for i in range(len(i0_w_names)):
        name = i0_w_names[i]
        i0_w = i0_weight[name]
        i0_g = i0_grad[name]
        i1_w = i1_weight[name]
        i2_w = i2_weight[name]
        diff = abs(i1_w - i0_w)
        print(name, np.max(diff))

def tpu_v_cpu():
    i0_cpu_weight  = np.load("cpu_weight_before_update_iter0.npz", allow_pickle=True)
    i0_cpu_w_names = [ k for k in i0_cpu_weight.keys()]
    i0_cpu_grad    = np.load("cpu_grad_iter0.npz", allow_pickle=True)
    i0_cpu_g_names = [ k for k in i0_cpu_grad.keys()]

    i0_tpu_weight  = np.load("model_weight_before_update_iter0.npz", allow_pickle=True)
    i0_tpu_w_names = [ k for k in i0_tpu_weight.keys()]
    i0_tpu_grad    = np.load("model_grad_iter0.npz", allow_pickle=True)
    i0_tpu_g_names = [ k for k in i0_tpu_grad.keys()]

    for i in range(len(i0_cpu_w_names)):
        name = i0_cpu_w_names[i]

        i0_w = i0_cpu_weight[name]
        i0_g = i0_cpu_grad[name]

        i1_w = i0_tpu_weight[name]
        i1_g = i0_tpu_grad[name]
        print(name, np.max(abs(i0_g-i1_g)))
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    #compare_weight("model_weight_epoch0.npz", "ema_weight_epoch1.npz")
    compare_grad("model_True_grad.npz","model_False_grad.npz")
    # i0_grad = np.load("tpu_iter0_model_grad.npz", allow_pickle=True)
    # i0_g_names = [ k for k in i0_grad.keys()]

    # i1_grad = np.load("tpu_iter1_model_grad.npz", allow_pickle=True)
    # i1_g_names = [ k for k in i1_grad.keys()]

    #test_cpu()
    #tpu_v_cpu()