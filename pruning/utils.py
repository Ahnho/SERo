import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

from .dpf.mnn import MaskLinear

def get_importance(weight, imp_type):
    if imp_type == 'L1':
        return weight.abs().mean(dim=1).detach().cpu().numpy()
    elif imp_type == 'L2':
        return weight.pow(2).mean(dim=1).detach().cpu().numpy()
    else:
        raise ValueError("Invalid importance type. Choose 'L1' or 'L2'.")
        
def expand_mask(mask, in_features):
    return np.repeat(mask[:, np.newaxis], in_features, axis=1)

def expand_value_mask(dim_mask, in_features, num_heads):
    head_masks = np.tile(dim_mask, (num_heads, 1))  
    
    flat_mask = head_masks.reshape(-1)  
    
    final_mask = np.tile(flat_mask[:, np.newaxis], (1, in_features))  
    
    return final_mask

def get_threshold(importance_all, rate):
    return np.percentile(importance_all, rate * 100)

def get_mask(importance, threshold):
    return importance > threshold
    # return importance < threshold


def get_aligned_mask_per_head(importance_per_head, threshold, target_dims):
    num_heads, head_dim = importance_per_head.shape
    masks = np.zeros((num_heads, head_dim), dtype=bool)
    
    for head_idx in range(num_heads):
        head_importance = importance_per_head[head_idx]
        sorted_idx = np.argsort(-head_importance)
        masks[head_idx, sorted_idx[:target_dims]] = True
        
    return masks

def get_aligned_target_dims(importance_per_head, threshold):
    kept_dims_per_head = (importance_per_head > threshold).sum(axis=1)
    avg_kept_dims = np.mean(kept_dims_per_head)
    
    if avg_kept_dims < 8:
        return 8
    power = int(np.ceil(np.log2(avg_kept_dims)))
    return max(8, 2 ** power)

def align_ffn_dims(dim_size, base=8):
    return ((dim_size + base - 1) // base) * base

def expand_mask(mask, in_features):
    return np.repeat(mask[:, np.newaxis], in_features, axis=1)

def new_get_vit_masks(model, pruning_rate, imp_type, mag_type):
    importance_dict = {'qk': [], 'v': [], 'ffn1': []}
    q_importance = None
    k_importance = None
    all_importance = []
    
    def process_attention_weights(weight_or_grad, imp_type):
        num_heads = weight_or_grad.size(0) // 64
        head_dim = 64
        weight_reshaped = weight_or_grad.view(num_heads, head_dim, -1)
        
        if imp_type == 'L1':
            return weight_reshaped.abs().mean(dim=2).detach().cpu().numpy()
        elif imp_type == 'L2':
            return weight_reshaped.pow(2).mean(dim=2).detach().cpu().numpy()
    
    def get_weight_or_grad(module, mag_type):
        return module.weight.grad if mag_type == 'grad' else module.weight
    
    # Collect importance scores
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            if "attention.attention" in name.lower():
                weight_or_grad = get_weight_or_grad(module, mag_type)
                
                if "query" in name:
                    q_importance = process_attention_weights(weight_or_grad, imp_type)
                elif "key" in name:
                    k_importance = process_attention_weights(weight_or_grad, imp_type)
                elif "value" in name:
                    v_importance = process_attention_weights(weight_or_grad, imp_type)
                    importance_dict['v'].append(v_importance)
                    all_importance.extend(v_importance.flatten())
                
                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict['qk'].append(qk_importance_avg)
                    all_importance.extend(qk_importance_avg.flatten())
                    q_importance = None
                    k_importance = None
                    
            elif "intermediate.dense" in name.lower():
                weight_or_grad = get_weight_or_grad(module, mag_type)
                ffn_importance = get_importance(weight_or_grad, imp_type)
                importance_dict['ffn1'].append(ffn_importance)
                all_importance.extend(ffn_importance.flatten())
    
    # Calculate threshold
    all_importance = np.array(all_importance)
    threshold = np.percentile(all_importance, pruning_rate * 100)
    
    # Generate masks
    masks = {}
    pruning_ratios = {'query': [], 'key': [], 'value': [], 'ffn1': []}
    dimension_stats = {'qk': [], 'v': [], 'ffn1': []}
    
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            if "attention.attention" in name.lower():
                if "query" in name:
                    head_importance = importance_dict['qk'].pop(0)
                    num_heads = head_importance.shape[0]
                    
                    target_dims = get_aligned_target_dims(head_importance, threshold)
                    head_masks = get_aligned_mask_per_head(head_importance, threshold, target_dims)
                    
                    flat_mask = head_masks.reshape(-1)
                    qk_mask = np.tile(flat_mask[:, np.newaxis], (1, module.in_features))
                    masks[f"{name}"] = qk_mask
                    masks[name.replace('query', 'key')] = qk_mask.copy()
                    
                    pruning_ratio = 1 - (np.sum(qk_mask) / qk_mask.size)
                    pruning_ratios['query'].append(pruning_ratio)
                    pruning_ratios['key'].append(pruning_ratio)
                    dimension_stats['qk'].append(target_dims)
                    
                elif "value" in name:
                    head_importance = importance_dict['v'].pop(0)
                    target_dims = get_aligned_target_dims(head_importance, threshold)
                    
                    head_masks = get_aligned_mask_per_head(head_importance, threshold, target_dims)
                    flat_mask = head_masks.reshape(-1)
                    
                    value_mask = np.tile(flat_mask[:, np.newaxis], (1, module.in_features))
                    masks[f"{name}"] = value_mask
                    
                    output_mask = np.tile(flat_mask[:, np.newaxis], (1, module.out_features)).T
                    masks[name.replace('attention.value', 'output.dense')] = output_mask
                    
                    pruning_ratio = 1 - (np.sum(value_mask) / value_mask.size)
                    pruning_ratios['value'].append(pruning_ratio)
                    dimension_stats['v'].append(target_dims)
                    
            elif "intermediate.dense" in name.lower():
                ffn_importance = importance_dict['ffn1'].pop(0)
                
                # FFN 차원을 8의 배수로 조정
                initial_mask = ffn_importance > threshold
                kept_dims = np.sum(initial_mask)
                target_dims = align_ffn_dims(kept_dims, base=8)
                
                # 중요도 순으로 정렬하여 상위 차원 선택
                sorted_idx = np.argsort(-ffn_importance)
                mask = np.zeros_like(ffn_importance, dtype=bool)
                mask[sorted_idx[:target_dims]] = True
                
                ffn_mask = expand_mask(mask, module.in_features)
                masks[f"{name}"] = ffn_mask
                
                output_name = name.replace('intermediate.dense', 'output.dense')
                output_module = next(m for n, m in model.named_modules() if n == output_name)
                masks[output_name] = expand_mask(mask, output_module.out_features).T
                
                pruning_ratio = 1 - (np.sum(ffn_mask) / ffn_mask.size)
                pruning_ratios['ffn1'].append(pruning_ratio)
                dimension_stats['ffn1'].append(target_dims)
    
    # # Print statistics
    # print("\nPruning Statistics:")
    # print("Dimensions kept per component:")
    # for key, dims in dimension_stats.items():
    #     if dims:
    #         avg_dims = sum(dims) / len(dims)
    #         print(f"{key}: {avg_dims:.1f} dimensions "
    #               f"(aligned to {8 if key == 'ffn1' else '2^n'}-multiple)")
    
    # print("\nPruning Ratios:")
    # for key, ratios in pruning_ratios.items():
    #     if ratios:
    #         avg_ratio = sum(ratios) / len(ratios)
    #         print(f"{key}: {avg_ratio:.4f} ({avg_ratio*100:.2f}%)")
    
    return masks

def get_independent_head_importance(weight_or_grad, imp_type):
    num_heads = weight_or_grad.size(0) // 64
    head_dim = 64
    weight_reshaped = weight_or_grad.view(num_heads, head_dim, -1)
    
    if imp_type == 'L1':
        return weight_reshaped.abs().mean(dim=2).detach().cpu().numpy()  # [num_heads, head_dim]
    elif imp_type == 'L2':
        return weight_reshaped.pow(2).mean(dim=2).detach().cpu().numpy()  # [num_heads, head_dim]

def generate_head_masks_with_min_dims(importance_per_head, threshold):
    num_heads, head_dim = importance_per_head.shape
    masks = []
    kept_dims = []
    
    for head_idx in range(num_heads):
        head_mask = importance_per_head[head_idx] > threshold
        kept_dim = max(1, np.sum(head_mask))  
        kept_dims.append(kept_dim)
        masks.append(head_mask)
    
    avg_kept_dims = int(np.mean(kept_dims))
    target_dims = max(1, avg_kept_dims)
    
    final_masks = np.zeros((num_heads, head_dim), dtype=bool)
    for head_idx in range(num_heads):
        head_importance = importance_per_head[head_idx]
        sorted_idx = np.argsort(-head_importance)
        final_masks[head_idx, sorted_idx[:target_dims]] = True
        
    return final_masks

def expand_independent_head_masks(head_masks, in_features):
    num_heads, head_dim = head_masks.shape
    flat_mask = head_masks.reshape(-1)
    return np.tile(flat_mask[:, np.newaxis], (1, in_features))

def get_vit_masks_with_independent_heads(model, pruning_rate, imp_type, mag_type):
    importance_dict = {'qk': [], 'v': [], 'ffn1': []}
    q_importance = None
    k_importance = None
    all_importance = []
    
    def get_weight_or_grad(module, mag_type):
        return module.weight.grad if mag_type == 'grad' else module.weight
    
    # Collect importance scores
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            weight_or_grad = get_weight_or_grad(module, mag_type)
            
            if "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_independent_head_importance(weight_or_grad, imp_type)
                elif "key" in name:
                    k_importance = get_independent_head_importance(weight_or_grad, imp_type)
                elif "value" in name:
                    v_importance = get_independent_head_importance(weight_or_grad, imp_type)
                    importance_dict['v'].append(v_importance)
                    all_importance.extend(v_importance.flatten())
                
                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict['qk'].append(qk_importance_avg)
                    all_importance.extend(qk_importance_avg.flatten())
                    q_importance = None
                    k_importance = None
            
            elif "intermediate.dense" in name.lower():
                ffn_importance = get_importance(weight_or_grad, imp_type)
                importance_dict['ffn1'].append(ffn_importance)
                all_importance.extend(ffn_importance.flatten())
    
    # Calculate threshold
    all_importance = np.array(all_importance)
    threshold = get_threshold(all_importance, pruning_rate)
    
    # Generate masks
    masks = {}
    pruning_ratios = {'query': [], 'key': [], 'value': [], 'ffn1': []}
    dimension_stats = {'qk': [], 'v': [], 'ffn1': []}
    
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear):
            if "attention.attention" in name.lower():
                if "query" in name:
                    head_importance = importance_dict['qk'].pop(0)
                    head_masks = generate_head_masks_with_min_dims(head_importance, threshold)
                    kept_dims = np.sum(head_masks, axis=1).mean()
                    
                    qk_mask = expand_independent_head_masks(head_masks, module.in_features)
                    masks[f"{name}"] = qk_mask
                    masks[name.replace('query', 'key')] = qk_mask.copy()
                    
                    pruning_ratio = 1 - (np.sum(qk_mask) / qk_mask.size)
                    pruning_ratios['query'].append(pruning_ratio)
                    pruning_ratios['key'].append(pruning_ratio)
                    dimension_stats['qk'].append(kept_dims)
                    
                elif "value" in name:
                    head_importance = importance_dict['v'].pop(0)
                    head_masks = generate_head_masks_with_min_dims(head_importance, threshold)
                    kept_dims = np.sum(head_masks, axis=1).mean()
                    
                    value_mask = expand_independent_head_masks(head_masks, module.in_features)
                    masks[f"{name}"] = value_mask
                    
                    output_mask = expand_independent_head_masks(head_masks, module.out_features).T
                    masks[name.replace('attention.value', 'output.dense')] = output_mask
                    
                    pruning_ratio = 1 - (np.sum(value_mask) / value_mask.size)
                    pruning_ratios['value'].append(pruning_ratio)
                    dimension_stats['v'].append(kept_dims)
                    
            elif "intermediate.dense" in name.lower():
                ffn_importance = importance_dict['ffn1'].pop(0)
                mask = ffn_importance > threshold
                kept_dims = max(1, np.sum(mask))
                
                sorted_idx = np.argsort(-ffn_importance)
                final_mask = np.zeros_like(mask)
                final_mask[sorted_idx[:kept_dims]] = True
                
                ffn_mask = expand_mask(final_mask, module.in_features)
                masks[f"{name}"] = ffn_mask
                
                output_name = name.replace('intermediate.dense', 'output.dense')
                output_module = next(m for n, m in model.named_modules() if n == output_name)
                masks[output_name] = expand_mask(final_mask, output_module.out_features).T
                
                pruning_ratio = 1 - (np.sum(ffn_mask) / ffn_mask.size)
                pruning_ratios['ffn1'].append(pruning_ratio)
                dimension_stats['ffn1'].append(kept_dims)
    
    # # Print statistics
    # print("\nPruning Statistics:")
    # print("Average dimensions kept per component:")
    # for key, dims in dimension_stats.items():
    #     if dims:
    #         avg_dims = sum(dims) / len(dims)
    #         print(f"{key}: {avg_dims:.1f} dimensions")
    
    # print("\nPruning Ratios:")
    # for key, ratios in pruning_ratios.items():
    #     if ratios:
    #         avg_ratio = sum(ratios) / len(ratios)
    #         print(f"{key}: {avg_ratio:.4f} ({avg_ratio*100:.2f}%)")
    
    return masks
    
def original_get_vit_masks(model, pruning_rate, imp_type, mag_type):
    importance_dict = {'qk': [], 'v': [], 'ffn1': []}
    q_importance = None
    k_importance = None
    all_importance = []
    
    if mag_type == 'weight':
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight, imp_type)
                elif "value" in name:
                    v_importance = get_importance(module.weight, imp_type)
                    importance_dict['v'].append(v_importance)
                    all_importance.append(v_importance)
                
                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict['qk'].append(qk_importance_avg)
                    all_importance.append(qk_importance_avg)
                    q_importance = None
                    k_importance = None
                    
            elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
                ffn_importance = get_importance(module.weight, imp_type)
                importance_dict['ffn1'].append(ffn_importance)
                all_importance.append(ffn_importance)

    elif mag_type == 'grad':
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight.grad, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight.grad, imp_type)
                elif "value" in name:
                    v_importance = get_importance(module.weight.grad, imp_type)
                    importance_dict['v'].append(v_importance)
                    all_importance.append(v_importance)
                
                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict['qk'].append(qk_importance_avg)
                    all_importance.append(qk_importance_avg)
                    q_importance = None
                    k_importance = None
                    
            elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
                ffn_importance = get_importance(module.weight.grad, imp_type)
                importance_dict['ffn1'].append(ffn_importance)
                all_importance.append(ffn_importance)

    # Calculate single threshold for all importances
    all_importance = np.concatenate(all_importance)
    threshold = get_threshold(all_importance, pruning_rate)
    
    # Generate masks
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
            if "query" in name :
                mask = importance_dict['qk'].pop(0) > threshold
                masks[f"{name}"] = expand_mask(mask, module.in_features)
            elif "value" in name:
                mask = importance_dict['v'].pop(0) > threshold
                masks[f"{name}"] = expand_mask(mask, module.in_features)
                # For attention output, we use the transpose of the value mask
                masks[name.replace('attention.value', 'output.dense')] = expand_mask(mask, module.out_features).T

        elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
            mask = importance_dict['ffn1'].pop(0) > threshold
            masks[f"{name}"] = expand_mask(mask, module.in_features)

            # For FFN output, we use the transpose of the intermediate mask
            output_name = name.replace('intermediate.dense', 'output.dense')
            output_module = next(m for n, m in model.named_modules() if n == output_name)
            masks[output_name] = expand_mask(mask, output_module.out_features).T

    return masks

def get_vit_masks(model, pruning_rate , imp_type, mag_type):
    importance_dict = {'qk': [], 'v': [], 'ffn1': []}
    q_importance = None
    k_importance = None
    
    if mag_type == 'weight':
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight, imp_type)
                elif "value" in name:
                    v_importance = get_importance(module.weight, imp_type)
                    importance_dict['v'].append(v_importance)
                
                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict['qk'].append(qk_importance_avg)
                    q_importance = None
                    k_importance = None
                    
            elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
                ffn_importance = get_importance(module.weight, imp_type)
                importance_dict['ffn1'].append(ffn_importance)

    elif mag_type == 'grad':
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight.grad, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight.grad, imp_type)
                elif "value" in name:
                    v_importance = get_importance(module.weight.grad, imp_type)
                    importance_dict['v'].append(v_importance)
                
                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict['qk'].append(qk_importance_avg)
                    q_importance = None
                    k_importance = None
                    
            elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
                ffn_importance = get_importance(module.weight.grad, imp_type)
                importance_dict['ffn1'].append(ffn_importance)
        

    # Calculate thresholds for each group
    thresholds = {}
    for key in importance_dict:
        if importance_dict[key]:
            importance_all = np.concatenate(importance_dict[key])
            # if key == 'v':
            #     thresholds[key] = 0
            # else:
            thresholds[key] = np.percentile(importance_all, pruning_rate * 100)
    
   # Generate masks
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
            if "query" in name :
                mask = importance_dict['qk'].pop(0) > thresholds['qk']
                masks[f"{name}"] = expand_mask(mask, module.in_features)
                # print(masks[f"{name}"].shape)
            elif "value" in name:
                mask = importance_dict['v'].pop(0) > thresholds['v']
                masks[f"{name}"] = expand_mask(mask, module.in_features)
                # For attention output, we use the transpose of the value mask
                masks[name.replace('attention.value', 'output.dense')] = expand_mask(mask, module.out_features).T
                # print(masks[f"{name}"].shape)

        elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
            mask = importance_dict['ffn1'].pop(0) > thresholds['ffn1']
            masks[f"{name}"] = expand_mask(mask, module.in_features)

            # For FFN output, we use the transpose of the intermediate mask
            output_name = name.replace('intermediate.dense', 'output.dense')
            output_module = next(m for n, m in model.named_modules() if n == output_name)
            masks[output_name] = expand_mask(mask, output_module.out_features).T
            # print(masks[output_name].shape)

    return masks
    
def apply_vit_masks(model, masks):
    for name, module in model.named_modules():
        with torch.no_grad():
            if isinstance(module, MaskLinear):  # 예: Linear 레이어만 선택
                if ("attention.attention.query" in name or "attention.attention.key" in name) :
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name :
                            mask_name = name.replace('key', 'query')
                            # print(mask_name)
                            param.data = torch.from_numpy(masks[mask_name]).float().cuda()

                elif "attention.attention.value" in name:
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name :
                            param.data = torch.from_numpy(masks[name]).float().cuda()
                            
                elif "attention.output" in name:
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name :
                            param.data = torch.from_numpy(masks[name]).float().cuda()

                elif "intermediate.dense" in name:
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name :
                            param.data = torch.from_numpy(masks[name]).float().cuda()
 
                elif "output.dense" in name and "attention" not in name:
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name :
                            param.data = torch.from_numpy(masks[name]).float().cuda()

                            
def vit_structured_pruning(model, pruning_rate, imp_type='L2'):
    masks = get_vit_masks(model, pruning_rate, imp_type)
    apply_vit_masks(model, masks)
    return model


def v_sparse_sum(model):
    state = model.state_dict()
    v_sparse_loss = 0.0
    for name, item in model.named_parameters():
        if  "attention.attention.value.weight" in name.lower():
        # if  "intermediate.dense.weight" in name.lower():
            m = name.replace('weight', 'mask')
           
            v_sparse_loss += (-1 * torch.log(1 +  (torch.abs(item) * state[m]))).mean() + torch.log(torch.tensor([2.0])).cuda()
            # total_sparse_loss += (torch.abs(item) * state[m]).mean()
            
    return v_sparse_loss

def cal_sparsity(model):
    mask_nonzeros = 0
    mask_length = 0
    total_weights = 0

    for name, item in model.named_parameters():
        if 'mask' in name and 'fc' not in name :
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.numel()

        # if 'weight' in name or 'bias' in name:
        if 'weight' in name and 'MaskLinear' in name and 'fc' not in name:
            total_weights += item.numel()

    num_zero = mask_length - mask_nonzeros
    sparsity = (num_zero / total_weights) * 100
    return total_weights, num_zero, sparsity

def cal_sparsity_nlp(model):
    mask_nonzeros = 0
    mask_length = 0
    total_weights = 0

    for name, item in model.named_parameters():
        if 'embeddings' not in name  and  'mask' in name:
            # print(name)
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.numel()

        # if 'weight' in name or 'bias' in name:
        if  'embeddings' not in name   and 'weight' in name:
            # print(name)
            total_weights += item.numel()

    num_zero = mask_length - mask_nonzeros
    sparsity = (num_zero / total_weights) * 100
    return total_weights, num_zero, sparsity

    
def original_get_vit_masks(model, pruning_rate, imp_type, mag_type):
    importance_dict = {'qk': [], 'v': [], 'ffn1': []}
    q_importance = None
    k_importance = None
    all_importance = []
    
    if mag_type == 'weight':
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight, imp_type)
                elif "value" in name:
                    v_importance = get_importance(module.weight, imp_type)
                    importance_dict['v'].append(v_importance)
                    all_importance.append(v_importance)
                
                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict['qk'].append(qk_importance_avg)
                    all_importance.append(qk_importance_avg)
                    q_importance = None
                    k_importance = None
                    
            elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
                ffn_importance = get_importance(module.weight, imp_type)
                importance_dict['ffn1'].append(ffn_importance)
                all_importance.append(ffn_importance)

    elif mag_type == 'grad':
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight.grad, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight.grad, imp_type)
                elif "value" in name:
                    v_importance = get_importance(module.weight.grad, imp_type)
                    importance_dict['v'].append(v_importance)
                    all_importance.append(v_importance)
                
                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict['qk'].append(qk_importance_avg)
                    all_importance.append(qk_importance_avg)
                    q_importance = None
                    k_importance = None
                    
            elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
                ffn_importance = get_importance(module.weight.grad, imp_type)
                importance_dict['ffn1'].append(ffn_importance)
                all_importance.append(ffn_importance)

    # Calculate single threshold for all importances
    all_importance = np.concatenate(all_importance)
    threshold = get_threshold(all_importance, pruning_rate)
    
    # Generate masks
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
            if "query" in name :
                mask = importance_dict['qk'].pop(0) > threshold
                masks[f"{name}"] = expand_mask(mask, module.in_features)
            elif "value" in name:
                mask = importance_dict['v'].pop(0) > threshold
                masks[f"{name}"] = expand_mask(mask, module.in_features)
                # For attention output, we use the transpose of the value mask
                masks[name.replace('attention.value', 'output.dense')] = expand_mask(mask, module.out_features).T

        elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
            mask = importance_dict['ffn1'].pop(0) > threshold
            masks[f"{name}"] = expand_mask(mask, module.in_features)

            # For FFN output, we use the transpose of the intermediate mask
            output_name = name.replace('intermediate.dense', 'output.dense')
            output_module = next(m for n, m in model.named_modules() if n == output_name)
            masks[output_name] = expand_mask(mask, output_module.out_features).T

    return masks

def get_vit_masks(model, pruning_rate , imp_type, mag_type):
    importance_dict = {'qk': [], 'v': [], 'ffn1': []}
    q_importance = None
    k_importance = None
    
    if mag_type == 'weight':
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight, imp_type)
                elif "value" in name:
                    v_importance = get_importance(module.weight, imp_type)
                    importance_dict['v'].append(v_importance)
                
                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict['qk'].append(qk_importance_avg)
                    q_importance = None
                    k_importance = None
                    
            elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
                ffn_importance = get_importance(module.weight, imp_type)
                importance_dict['ffn1'].append(ffn_importance)

    elif mag_type == 'grad':
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight.grad, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight.grad, imp_type)
                elif "value" in name:
                    v_importance = get_importance(module.weight.grad, imp_type)
                    importance_dict['v'].append(v_importance)
                
                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict['qk'].append(qk_importance_avg)
                    q_importance = None
                    k_importance = None
                    
            elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
                ffn_importance = get_importance(module.weight.grad, imp_type)
                importance_dict['ffn1'].append(ffn_importance)
        

    # Calculate thresholds for each group
    thresholds = {}
    for key in importance_dict:
        if importance_dict[key]:
            importance_all = np.concatenate(importance_dict[key])
            # if key == 'v':
            #     thresholds[key] = 0
            # else:
            thresholds[key] = np.percentile(importance_all, pruning_rate * 100)
    
   # Generate masks
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
            if "query" in name :
                mask = importance_dict['qk'].pop(0) > thresholds['qk']
                masks[f"{name}"] = expand_mask(mask, module.in_features)
                # print(masks[f"{name}"].shape)
            elif "value" in name:
                mask = importance_dict['v'].pop(0) > thresholds['v']
                masks[f"{name}"] = expand_mask(mask, module.in_features)
                # For attention output, we use the transpose of the value mask
                masks[name.replace('attention.value', 'output.dense')] = expand_mask(mask, module.out_features).T
                # print(masks[f"{name}"].shape)

        elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
            mask = importance_dict['ffn1'].pop(0) > thresholds['ffn1']
            masks[f"{name}"] = expand_mask(mask, module.in_features)

            # For FFN output, we use the transpose of the intermediate mask
            output_name = name.replace('intermediate.dense', 'output.dense')
            output_module = next(m for n, m in model.named_modules() if n == output_name)
            masks[output_name] = expand_mask(mask, output_module.out_features).T
            # print(masks[output_name].shape)

    return masks
    
def apply_vit_masks(model, masks):
    for name, module in model.named_modules():
        with torch.no_grad():
            if isinstance(module, MaskLinear):  # 예: Linear 레이어만 선택
                if ("attention.attention.query" in name or "attention.attention.key" in name) :
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name :
                            mask_name = name.replace('key', 'query')
                            # print(mask_name)
                            param.data = torch.from_numpy(masks[mask_name]).float().cuda()

                elif "attention.attention.value" in name:
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name :
                            param.data = torch.from_numpy(masks[name]).float().cuda()
                            
                elif "attention.output" in name:
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name :
                            param.data = torch.from_numpy(masks[name]).float().cuda()

                elif "intermediate.dense" in name:
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name :
                            param.data = torch.from_numpy(masks[name]).float().cuda()
 
                elif "output.dense" in name and "attention" not in name:
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name :
                            param.data = torch.from_numpy(masks[name]).float().cuda()

                            
def vit_structured_pruning(model, pruning_rate, imp_type='L2'):
    masks = get_vit_masks(model, pruning_rate, imp_type)
    apply_vit_masks(model, masks)
    return model


def v_sparse_sum(model):
    state = model.state_dict()
    v_sparse_loss = 0.0
    for name, item in model.named_parameters():
        if  "attention.attention.value.weight" in name.lower():
        # if  "intermediate.dense.weight" in name.lower():
            m = name.replace('weight', 'mask')
           
            v_sparse_loss += (-1 * torch.log(1 +  (torch.abs(item) * state[m]))).mean() + torch.log(torch.tensor([2.0])).cuda()
            # total_sparse_loss += (torch.abs(item) * state[m]).mean()
            
    return v_sparse_loss

def cal_sparsity(model):
    mask_nonzeros = 0
    mask_length = 0
    total_weights = 0

    for name, item in model.named_parameters():
        if 'mask' in name and 'fc' not in name :
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.numel()

        # if 'weight' in name or 'bias' in name:
        if 'weight' in name and 'MaskLinear' in name and 'fc' not in name:
            total_weights += item.numel()

    num_zero = mask_length - mask_nonzeros
    sparsity = (num_zero / total_weights) * 100
    return total_weights, num_zero, sparsity

def cal_sparsity_nlp(model):
    mask_nonzeros = 0
    mask_length = 0
    total_weights = 0

    for name, item in model.named_parameters():
        if 'embeddings' not in name  and  'mask' in name:
            # print(name)
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.numel()

        # if 'weight' in name or 'bias' in name:
        if  'embeddings' not in name   and 'weight' in name:
            # print(name)
            total_weights += item.numel()

    num_zero = mask_length - mask_nonzeros
    sparsity = (num_zero / total_weights) * 100
    return total_weights, num_zero, sparsity
