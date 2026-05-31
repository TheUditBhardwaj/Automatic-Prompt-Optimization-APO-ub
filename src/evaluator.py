import logging
import itertools
from typing import Dict, Any, Union, List, Tuple, Optional

logger = logging.getLogger(__name__)

def flatten_dict(d: Union[Dict[str, Any], List[Any], Any], prefix: str = "") -> Dict[str, Any]:
    """Flattens a nested dictionary/list structure into a flat key-value map.
    
    Nested paths are separated by periods. Lists use index numbers as path parts.
    """
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            items.update(flatten_dict(v, new_key))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{prefix}.{i}" if prefix else str(i)
            items.update(flatten_dict(v, new_key))
    else:
        # Leaf nodes (string, number, bool, None)
        items[prefix] = d
    return items

def compute_item_similarity(item1: Any, item2: Any) -> int:
    """Counts matching primitive field values between two items."""
    flat1 = flatten_dict(item1)
    flat2 = flatten_dict(item2)
    matches = 0
    for k, v in flat2.items():
        if v in (None, "", [], {}):
            continue
        if k in flat1:
            val1 = flat1[k]
            if isinstance(v, str) and isinstance(val1, str):
                if v.strip().lower() == val1.strip().lower():
                    matches += 1
            elif v == val1:
                matches += 1
    return matches

def align_array_fields(pred_list: List[Any], gold_list: List[Any]) -> List[Any]:
    """Finds the best matching alignment of predicted items to gold items.
    
    Uses a greedy matching algorithm to avoid combinatorial explosion (O(N!) -> O(N^2)).
    """
    if not pred_list or not gold_list:
        return pred_list

    n_pred = len(pred_list)
    n_gold = len(gold_list)
    
    # Compute similarity matrix
    similarity_matrix = []
    for pred_idx in range(n_pred):
        row = []
        for gold_idx in range(n_gold):
            sim = compute_item_similarity(pred_list[pred_idx], gold_list[gold_idx])
            row.append((sim, pred_idx, gold_idx))
        similarity_matrix.append(row)
        
    # Greedy assignment
    mapped_preds = set()
    mapped_golds = set()
    best_mapping = {}  # pred_idx -> gold_idx
    
    # Flatten and sort all possible matches by similarity score descending
    all_matches = []
    for row in similarity_matrix:
        all_matches.extend(row)
    all_matches.sort(key=lambda x: x[0], reverse=True)
    
    for sim, pred_idx, gold_idx in all_matches:
        if pred_idx not in mapped_preds and gold_idx not in mapped_golds:
            best_mapping[pred_idx] = gold_idx
            mapped_preds.add(pred_idx)
            mapped_golds.add(gold_idx)
            
    # Reconstruct prediction list aligned with gold list indices
    aligned_pred = [None] * n_gold
    unmapped_preds = []
    
    for pred_idx in range(n_pred):
        if pred_idx in best_mapping:
            gold_idx = best_mapping[pred_idx]
            aligned_pred[gold_idx] = pred_list[pred_idx]
        else:
            unmapped_preds.append(pred_list[pred_idx])
            
    # Append any remaining unmapped predictions
    aligned_pred = [x for x in aligned_pred if x is not None] + unmapped_preds
    return aligned_pred


def normalize_skills(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes predicted Pydantic SkillGroups to match the gold JSON skills format."""
    normalized = dict(pred)
    if "skills" in normalized and isinstance(normalized["skills"], list):
        skills_dict = {}
        for group in normalized["skills"]:
            category = group.get("category") if isinstance(group, dict) else getattr(group, "category", None)
            items = group.get("items") if isinstance(group, dict) else getattr(group, "items", None)
            if category is not None and items is not None:
                skills_dict[category] = items
        
        # Match structure to gold skills
        if "skills" in gold:
            if isinstance(gold["skills"], list):
                # Flatten dictionary lists into a single flat list
                flat_skills = []
                for s_list in skills_dict.values():
                    flat_skills.extend(s_list)
                normalized["skills"] = flat_skills
            elif isinstance(gold["skills"], dict):
                normalized["skills"] = skills_dict
            elif gold["skills"] is None:
                normalized["skills"] = None
                
    return normalized

def align_and_sort_pred(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    """Aligns lists of objects and sorts lists of primitives between pred and gold."""
    aligned = dict(pred)
    
    for key, gold_val in gold.items():
        if key not in aligned:
            continue
            
        pred_val = aligned[key]
        
        # Align lists of dicts (e.g. workExperience, education, publications)
        if isinstance(gold_val, list) and isinstance(pred_val, list):
            if len(gold_val) > 0 and isinstance(gold_val[0], dict):
                aligned[key] = align_array_fields(pred_val, gold_val)
            else:
                # Sort lists of primitives (e.g. skills list, languages, socialLinks)
                try:
                    aligned[key] = sorted(pred_val, key=lambda x: str(x).lower())
                except Exception:
                    pass
                    
    return aligned

def filter_meaningful_fields(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Removes null, empty, or default empty structures from flattened dict."""
    filtered = {}
    for k, v in flat_dict.items():
        if v in (None, "", [], {}):
            continue
        filtered[k] = v
    return filtered

import difflib

def parse_evaluation_config(config: Any) -> Tuple[str, Dict[str, Any]]:
    """Parses evaluation_config format (either string or dictionary with metrics list)."""
    if isinstance(config, str):
        return config, {}
    if isinstance(config, dict):
        metrics = config.get("metrics", [])
        if metrics:
            metric = metrics[0]
            return metric.get("metric_id", ""), metric.get("params", {})
    return "", {}

def get_evaluation_config_for_path(path: str, schema_dict: Dict[str, Any]) -> Any:
    """Recursively walks the JSON Schema to retrieve the evaluation_config for a dot-separated path."""
    parts = path.split('.')
    
    def resolve_node(node: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref_path = node["$ref"]
            if ref_path.startswith("#/"):
                ref_parts = ref_path.split("/")[1:]
                current = schema_dict
                for p in ref_parts:
                    if p in current:
                        current = current[p]
                    else:
                        return node
                return resolve_node(current)
        return node

    current = resolve_node(schema_dict)
    
    for part in parts:
        current = resolve_node(current)
        
        # Handle anyOf / oneOf
        if "anyOf" in current or "oneOf" in current:
            options = current.get("anyOf") or current.get("oneOf")
            found = False
            for option in options:
                opt_resolved = resolve_node(option)
                if isinstance(opt_resolved, dict):
                    if part.isdigit() and opt_resolved.get("type") == "array":
                        current = opt_resolved
                        found = True
                        break
                    elif not part.isdigit() and "properties" in opt_resolved and part in opt_resolved["properties"]:
                        current = opt_resolved
                        found = True
                        break
            if not found:
                for option in options:
                    opt_resolved = resolve_node(option)
                    if isinstance(opt_resolved, dict):
                        if part.isdigit() and "items" in opt_resolved:
                            current = opt_resolved
                            break
                        elif not part.isdigit() and "properties" in opt_resolved:
                            current = opt_resolved
                            break
                            
        current = resolve_node(current)
        
        if part.isdigit():
            if "items" in current:
                current = resolve_node(current["items"])
        else:
            if "properties" in current and part in current["properties"]:
                current = resolve_node(current["properties"][part])
            elif "additionalProperties" in current and isinstance(current["additionalProperties"], dict):
                current = resolve_node(current["additionalProperties"])
                
    current = resolve_node(current)
    if isinstance(current, dict):
        return current.get("evaluation_config")
    return None

def check_semantic_match_cached(pred: str, gold: str, client: Optional[Any] = None, db: Optional[Any] = None) -> bool:
    """Checks semantic equivalence using Gemini (fixed parameters) and persistent database caching."""
    p_norm = pred.strip()
    g_norm = gold.strip()
    
    if p_norm.lower() == g_norm.lower():
        return True
        
    # Check DB
    if db is not None:
        try:
            cached = db.get_semantic_match(p_norm, g_norm)
            if cached is not None:
                return cached
        except Exception as e:
            logger.debug(f"DB semantic cache fetch failed: {e}")
            
    # Check memory cache
    if not hasattr(check_semantic_match_cached, "_mem_cache"):
        check_semantic_match_cached._mem_cache = {}
    cache_key = (p_norm, g_norm)
    if cache_key in check_semantic_match_cached._mem_cache:
        return check_semantic_match_cached._mem_cache[cache_key]
        
    is_match = False
    # Set to False to bypass calling the Gemini API for semantic evaluation and use local Jaccard similarity fallback instead.
    # This prevents running out of API quota quickly.
    USE_API_FOR_SEMANTIC_EVAL = False
    if client is not None and USE_API_FOR_SEMANTIC_EVAL:
        try:
            is_match = client.check_semantic_similarity(p_norm, g_norm)
        except Exception as e:
            logger.warning(f"Gemini semantic similarity check failed: {e}. Falling back to Jaccard.")
            is_match = local_jaccard_similarity(p_norm, g_norm) >= 0.7
    else:
        is_match = local_jaccard_similarity(p_norm, g_norm) >= 0.7
        
    # Save cache
    check_semantic_match_cached._mem_cache[cache_key] = is_match
    if db is not None:
        try:
            db.save_semantic_match(p_norm, g_norm, is_match)
        except Exception as e:
            logger.debug(f"DB semantic cache save failed: {e}")
            
    return is_match

def local_jaccard_similarity(s1: str, s2: str) -> float:
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    if not w1 and not w2:
        return 1.0
    return len(w1.intersection(w2)) / len(w1.union(w2))

def evaluate_field_match(
    path: str, 
    pred: Any, 
    gold: Any, 
    schema_dict: Optional[Dict[str, Any]] = None, 
    client: Optional[Any] = None, 
    db: Optional[Any] = None
) -> bool:
    """Evaluates field-level matching by honoring schema evaluation configurations."""
    if schema_dict is None:
        return str(pred).strip().lower() == str(gold).strip().lower()
        
    config = get_evaluation_config_for_path(path, schema_dict)
    metric_id, params = parse_evaluation_config(config)
    
    if not metric_id:
        metric_id = "string_case_insensitive"
        
    try:
        if metric_id == "string_exact":
            return str(pred) == str(gold)
            
        elif metric_id == "string_case_insensitive":
            return str(pred).strip().lower() == str(gold).strip().lower()
            
        elif metric_id == "string_fuzzy":
            ratio = difflib.SequenceMatcher(None, str(pred).strip().lower(), str(gold).strip().lower()).ratio()
            threshold = params.get("threshold", 0.8)
            return ratio >= threshold
            
        elif metric_id == "string_semantic":
            return check_semantic_match_cached(str(pred), str(gold), client, db)
            
        elif metric_id == "integer_exact":
            return int(float(pred)) == int(float(gold))
            
        elif metric_id == "number_exact":
            return float(pred) == float(gold)
            
        elif metric_id == "number_tolerance":
            tolerance = params.get("tolerance", 1e-3)
            return abs(float(pred) - float(gold)) <= tolerance
            
        elif metric_id == "boolean_exact":
            def to_bool(v):
                if isinstance(v, bool):
                    return v
                return str(v).lower() in ("true", "1", "yes")
            return to_bool(pred) == to_bool(gold)
            
        else:
            return str(pred).strip().lower() == str(gold).strip().lower()
    except Exception as e:
        logger.debug(f"Comparison error for field {path} using metric {metric_id}: {e}")
        return False

def compute_precision_recall_f1(
    pred_flat: Dict[str, Any], 
    gold_flat: Dict[str, Any], 
    schema_dict: Optional[Dict[str, Any]] = None,
    client: Optional[Any] = None,
    db: Optional[Any] = None
) -> Tuple[float, float, float, int, int, int]:
    """Computes Precision, Recall, F1 and counts using dynamic per-field evaluation metrics."""
    pred_clean = filter_meaningful_fields(pred_flat)
    gold_clean = filter_meaningful_fields(gold_flat)
    
    if not gold_clean:
        if not pred_clean:
            return 1.0, 1.0, 1.0, 0, 0, 0
        else:
            return 0.0, 1.0, 0.0, 0, len(pred_clean), 0

    tp = 0
    fp = 0
    fn = 0
    
    for k, g_val in gold_clean.items():
        if k in pred_clean:
            p_val = pred_clean[k]
            is_match = evaluate_field_match(k, p_val, g_val, schema_dict, client, db)
            if is_match:
                tp += 1
            else:
                fp += 1
                fn += 1
        else:
            fn += 1
            
    for k in pred_clean.keys():
        if k not in gold_clean:
            fp += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1, tp, fp, fn

def get_schema_path(path: str) -> str:
    """Helper to convert a concrete path like workExperience.0.employer to workExperience.employer."""
    parts = path.split('.')
    new_parts = [p for p in parts if not p.isdigit()]
    return ".".join(new_parts)

def evaluate_predictions(
    pred: Dict[str, Any], 
    gold: Dict[str, Any], 
    schema_dict: Optional[Dict[str, Any]] = None,
    client: Optional[Any] = None,
    db: Optional[Any] = None
) -> Dict[str, Any]:
    """Computes structural Precision, Recall, F1 globally and per top-level field."""
    # 1. Format and normalize skills
    normalized_pred = normalize_skills(pred, gold)
    
    # 2. Align and sort lists
    aligned_pred = align_and_sort_pred(normalized_pred, gold)
    
    # Flatten full dictionaries
    flat_pred = flatten_dict(aligned_pred)
    flat_gold = flatten_dict(gold)
    
    # Compute global score
    p, r, f1, tp, fp, fn = compute_precision_recall_f1(flat_pred, flat_gold, schema_dict, client, db)
    
    # Compute per-field (subtree) scores
    per_field = {}
    top_level_keys = set(gold.keys()).union(set(pred.keys()))
    for key in top_level_keys:
        gold_sub = gold.get(key)
        pred_sub = aligned_pred.get(key)
        
        flat_gold_sub = flatten_dict({key: gold_sub}) if gold_sub is not None else {}
        flat_pred_sub = flatten_dict({key: pred_sub}) if pred_sub is not None else {}
        
        sub_p, sub_r, sub_f1, _, _, _ = compute_precision_recall_f1(flat_pred_sub, flat_gold_sub, schema_dict, client, db)
        per_field[key] = {
            "precision": sub_p,
            "recall": sub_r,
            "f1": sub_f1
        }
        
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "per_field": per_field
    }

def aggregate_run_metrics(
    all_preds: List[Dict[str, Any]], 
    all_golds: List[Dict[str, Any]], 
    schema_dict: Dict[str, Any],
    client: Optional[Any] = None,
    db: Optional[Any] = None
) -> Dict[str, Any]:
    """Computes global, per-subtree, and per-leaf metrics aggregated across multiple documents."""
    leaf_stats = {}
    
    for pred, gold in zip(all_preds, all_golds):
        normalized_pred = normalize_skills(pred, gold)
        aligned_pred = align_and_sort_pred(normalized_pred, gold)
        
        flat_pred = filter_meaningful_fields(flatten_dict(aligned_pred))
        flat_gold = filter_meaningful_fields(flatten_dict(gold))
        
        all_paths = set(flat_pred.keys()).union(set(flat_gold.keys()))
        for path in all_paths:
            schema_path = get_schema_path(path)
            if schema_path not in leaf_stats:
                leaf_stats[schema_path] = {"tp": 0, "fp": 0, "fn": 0}
                
            if path in flat_gold:
                if path in flat_pred:
                    is_match = evaluate_field_match(path, flat_pred[path], flat_gold[path], schema_dict, client, db)
                    if is_match:
                        leaf_stats[schema_path]["tp"] += 1
                    else:
                        leaf_stats[schema_path]["fp"] += 1
                        leaf_stats[schema_path]["fn"] += 1
                else:
                    leaf_stats[schema_path]["fn"] += 1
            else:
                if path in flat_pred:
                    leaf_stats[schema_path]["fp"] += 1
                    
    # Compute precision, recall, F1 per leaf
    per_leaf = {}
    for spath, stats in leaf_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        per_leaf[spath] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        
    # Aggregate per subtree (top-level key)
    subtree_stats = {}
    for spath, stats in leaf_stats.items():
        top_level = spath.split('.')[0]
        if top_level not in subtree_stats:
            subtree_stats[top_level] = {"tp": 0, "fp": 0, "fn": 0}
        subtree_stats[top_level]["tp"] += stats["tp"]
        subtree_stats[top_level]["fp"] += stats["fp"]
        subtree_stats[top_level]["fn"] += stats["fn"]
        
    per_subtree = {}
    for sub, stats in subtree_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        per_subtree[sub] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        
    # Global aggregation
    global_tp = sum(s["tp"] for s in leaf_stats.values())
    global_fp = sum(s["fp"] for s in leaf_stats.values())
    global_fn = sum(s["fn"] for s in leaf_stats.values())
    
    global_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    global_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    global_f1 = (2 * global_p * global_r) / (global_p + global_r) if (global_p + global_r) > 0 else 0.0
    
    return {
        "global": {
            "precision": global_p,
            "recall": global_r,
            "f1": global_f1,
            "tp": global_tp,
            "fp": global_fp,
            "fn": global_fn
        },
        "per_subtree": per_subtree,
        "per_leaf": per_leaf
    }

def compute_exact_match_score(
    pred: Dict[str, Any], 
    gold: Dict[str, Any], 
    schema_dict: Optional[Dict[str, Any]] = None,
    client: Optional[Any] = None,
    db: Optional[Any] = None
) -> float:
    """Kept for backward compatibility with baseline main runner. Returns global F1 score."""
    scores = evaluate_predictions(pred, gold, schema_dict, client, db)
    return scores["f1"]

