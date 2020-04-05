import numpy as np

def get_image_name(image_path):
    return '/'.join(image_path.split('/')[-2:])

def check_model_weights(idx, model):

    params = list(model.parameters())
    diagnostic = None
    diagnostic_grad = None
    for layer_num, param in enumerate(params[0:1]):
        diagnostic = param.detach().cpu().numpy().flatten()[10:13]
        diagnostic = " ".join([str(x)[:7] for x in diagnostic.tolist()])
        if param.grad is not None:
            diagnostic_grad = param.grad.detach().cpu().numpy().flatten()[10:13]
            diagnostic_grad = " ".join([str(x)[:7] for x in diagnostic_grad.tolist()])
            print(idx, '\t', diagnostic, ';', diagnostic_grad)
        else:
            print(idx, '\t', diagnostic, '; param.grad does not exist')


######################################################
# just fetch the magnitude of the gradient

def model_to_grad_as_dict(model):
    grad_as_dict = {}
    params = list(model.parameters())
    for layer_num, param in enumerate(params):
        layer_grad_np = param.grad.clone()
        grad_as_dict[layer_num] = layer_grad_np
    return grad_as_dict

def get_flatten_grad(grad_as_dict, grad_layers):
    if len(grad_layers) == 1 and grad_layers[0] == -1:
        return np.concatenate([v.detach().cpu().numpy().flatten() for k, v in grad_as_dict.items()])
    else:
        return np.concatenate([v.detach().cpu().numpy().flatten() for k, v in grad_as_dict.items() if k in grad_layers])

def model_to_grad_as_dict_and_flatten(model, grad_layers):
    grad_as_dict = model_to_grad_as_dict(model)
    grad_flattened = get_flatten_grad(grad_as_dict, grad_layers)
    return grad_as_dict, grad_flattened



######################################################
# Weighting and ranking

def get_agreement(arr_1, arr_2):
    return np.dot(arr_1, arr_2)

def get_agreements_dict(current_fake_batch_grad_dict):

    idx_ij_to_agreement = {}
    keys = list(sorted(current_fake_batch_grad_dict.keys()))
    
    for i in keys:
        for j in keys:
            if j > i:
                grad_flatten_i = current_fake_batch_grad_dict[i][1]
                grad_flatten_j = current_fake_batch_grad_dict[j][1]
                agreement_i_j = get_agreement(grad_flatten_i, grad_flatten_j)
                idx_ij_to_agreement[f"{i};{j}"] = agreement_i_j
    
    return idx_ij_to_agreement

def get_softmax_single_value(x, annealling_factor):
    return np.exp(x * annealling_factor)

def get_softmax_denominator(d, annealling_factor):
    agreements_np = np.array([v for v in d.values()])
    agreements_log = np.exp(agreements_np * annealling_factor)
    softmax_denominator = np.sum(agreements_log)
    return softmax_denominator

def apply_softmax_to_dict(d, annealling_factor):
    softmax_denominator = get_softmax_denominator(d, annealling_factor)
    d_softmax = {}
    for k, v in d.items():
        d_softmax[k] = get_softmax_single_value(v, annealling_factor) / softmax_denominator
    return d_softmax

def get_relevant_keys(d, idx):
    relevant_keys = []
    for k in d.keys():
        items = k.split(';')
        idx_1 = int(items[0])
        idx_2 = int(items[1])
        if idx_1 == idx or idx_2 == idx:
            relevant_keys.append(k)
    return relevant_keys

def get_idx_agreement_sum(d, idx):
    relevant_keys = get_relevant_keys(d, idx)
    relevant_values = [v for k, v in d.items() if k in relevant_keys]
    idx_agreement_sum = sum(relevant_values)
    return idx_agreement_sum

def get_idx_to_weight(current_fake_batch_grad_dict, annealling_factor, idx_to_gt):
    if len(current_fake_batch_grad_dict) == 1:
        return {k: 1.0 for k in current_fake_batch_grad_dict.keys()}
    idx_ij_to_agreement = get_agreements_dict(current_fake_batch_grad_dict)
    idx_ij_to_agreement_softmax = apply_softmax_to_dict(idx_ij_to_agreement, annealling_factor)
    pos_same_class_count, neg_same_class_count = 0, 0
    pos_different_class_count, neg_different_class_count = 0, 0
    for k, v in idx_ij_to_agreement.items():
        parts = k.split(';')
        idx_1 = int(parts[0])
        idx_2 = int(parts[1])
        if idx_to_gt[idx_1] == idx_to_gt[idx_2]:
            if v > 0:
                pos_same_class_count += 1
            else:
                neg_same_class_count += 1
        else:
            if v > 0:
                pos_different_class_count += 1
            else:
                neg_different_class_count += 1
    # print(pos_same_class_count, neg_same_class_count, pos_different_class_count, neg_different_class_count)
    idx_to_weight = {}
    for idx in current_fake_batch_grad_dict.keys():
        idx_agreement_sum = get_idx_agreement_sum(idx_ij_to_agreement_softmax, idx)
        idx_to_weight[idx] = idx_agreement_sum / 2

    return idx_to_weight



######################################################
# update model using the weighted gradient

def get_new_layer_grad(layer_num, idx_to_weight_batch, minibatch_grad_dict):


    indices = list(idx_to_weight_batch.keys())
    initial_idx = indices[0]
    weighted_grad = idx_to_weight_batch[initial_idx] * minibatch_grad_dict[initial_idx][0][layer_num]
    for idx in indices[1:]:
        grad_to_add_idx = idx_to_weight_batch[idx] * minibatch_grad_dict[idx][0][layer_num]
        weighted_grad = weighted_grad.add(grad_to_add_idx)
    return weighted_grad