variant_labels = {
    # "split-curric": "Split",
    "split_120-curric": "Split",
}

leaf_labels = {
    "pred_sem": "Predict Sem",
    "gt_sem": "GT Sem",
}

def get_variant_labels(variant, labels=variant_labels, is_gt=False):
    stem_leaf = variant.split('/')
    stem = stem_leaf[0]
    leaf = "/".join(stem_leaf[1:])
    stem_label = labels.get(stem, variant)
    if len(leaf) == 0:
        leaf_label = leaf_labels['gt_sem' if is_gt else 'pred_sem']
    else:
        leaf_label =  leaf_labels.get(leaf, leaf)
    return f"{stem_label} - {leaf_label}"

key_labels = {  "spl": "SPL - Train",
                "success": "Success - Train",
                "eval_spl": "SPL - Val",
                "eval_success": "Success - Val",
                "eval_coverage": "Coverage - Val",
                "eval_steps": "Steps - Val",
                'eval_distance_to_goal': "D2G - Val"
            }
axis_labels = {
    "spl": "SPL",
    "eval_spl": "SPL",
    "success": "Success",
    "eval_success": "Success",
    "eval_coverage": "Coverage"
}

plot_key_folder_dict = {
    'eval_spl': 'eval_metrics_spl/',
    'eval_success': 'eval_metrics_success/',
    'eval_distance_to_goal': 'eval_metrics_distance_to_goal/',
    'eval_coverage': 'eval_metrics_coverage.reached/',
    'eval_steps': 'eval_metrics_coverage.step/',
    'eval_': 'eval_metrics_coverage.steps/',
}

def get_obj_label(key):
    words = key.split("_")
    return " ".join(map(lambda w: str(w[0]).upper() + w[1:], words))
