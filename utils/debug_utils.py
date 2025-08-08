import torch
import numpy as np

def get_model_grad(model, save_path="model_grad.npz", save_ = True):
    grad_dict = {}  
    for name, param in model.named_parameters():
        if isinstance(param.grad, torch.Tensor):
            n_name = name
            #n_name = name + "_" + hex(param.grad.data_ptr())
            #n_name = hex(param.grad.data_ptr())
            grad_dict[n_name] = param.grad.cpu().numpy()
        else:
            print(name, "has no grad")
    if save_:
        np.savez(save_path, **grad_dict)
    return grad_dict

def get_model_weight(model, save_path="model_grad.npz", save_ = True):
    grad_dict = {}  
    for name, param in model.named_parameters():
        if isinstance(param.data, torch.Tensor):
            n_name  = name
            #n_name = name + "_" + hex(param.data.data_ptr())
            #n_name = hex(param.data.data_ptr())
            grad_dict[n_name] = param.data.cpu().numpy()
        else:
            print(name, "has no grad")
    if save_:
        np.savez(save_path, **grad_dict)
    return grad_dict