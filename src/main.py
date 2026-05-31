import os
import argparse
import json
import logging
from datetime import datetime
from typing import List

from config import Config
from schema import get_schema_class, load_schema_dict
from dataset import load_dataset, split_dataset, DatasetItem
from pdf_loader import extract_text_from_pdf
from prompts import DEFAULT_SYSTEM_INSTRUCTION, get_seed_prompt, format_extraction_prompt
from llm_wrapper import GeminiClient
from parser import parse_json_to_schema
from evaluator import evaluate_predictions, compute_exact_match_score, aggregate_run_metrics
from database import APODatabase

# Setup clean logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def run_baseline(config_path: str):
    """Runs baseline structured extraction over the validation set with SQLite logging."""
    logger.info("Initializing baseline extraction run...")
    
    # 1. Load configuration
    config = Config(config_path)
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Target Model: {config.model_name}")
    logger.info(f"Target Schema: {config.schema_type}")

    # 2. Load dataset
    all_items = load_dataset(config.raw_dir, config.gold_dir)
    if not all_items:
        raise ValueError(
            f"No matching PDF/JSON data pairs found in {config.raw_dir} and {config.gold_dir}. "
            "Please run 'python3 src/generate_mock_data.py' to generate mock data."
        )

    # 3. Deterministic Split
    train_items, val_items, test_items = split_dataset(
        items=all_items,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed
    )
    
    logger.info(f"Train split items: {[item.name for item in train_items]}")
    logger.info(f"Val split items: {[item.name for item in val_items]}")
    logger.info(f"Test split items: {[item.name for item in test_items]}")

    if not val_items:
        logger.warning("Validation split is empty! Running baseline on the entire dataset instead.")
        eval_items = all_items
    else:
        eval_items = val_items

    # 4. Initialize Gemini client and SQLite DB
    try:
        client = GeminiClient()
    except Exception as e:
        logger.error("Failed to initialize Gemini Client. Please check your configuration.")
        return
        
    db = APODatabase()

    # 5. Prepare run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"baseline_{timestamp}"
    run_dir = os.path.join(config.output_dir, run_id)
    preds_dir = os.path.join(run_dir, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

    # Save copy of the run config
    with open(os.path.join(run_dir, "config_run.yaml"), "w") as f:
        json.dump(config.data, f, indent=4)

    # Register Run in DB
    db.save_run(
        run_id=run_id,
        timestamp=timestamp,
        model_name=config.model_name,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        config_dict=config.data
    )

    # 6. Extraction Loop
    schema_class = get_schema_class(config.schema_type)
    schema_dict = load_schema_dict(config.schema_type)
    seed_prompt = get_seed_prompt(config.schema_type, config.seed_prompt)
    
    results = []
    total_score = 0.0
    total_prompt_tokens = 0
    total_candidate_tokens = 0
    total_cost = 0.0

    print("\n" + "="*60)
    print(f"STARTING BASELINE RUN ON {len(eval_items)} ITEMS (Val Split) | Run ID: {run_id}")
    print("="*60)

    for idx, item in enumerate(eval_items, 1):
        print(f"[{idx}/{len(eval_items)}] Extracting: {item.name}.pdf ...")
        gold_data = item.load_gold_json()
        
        # 6a. Check Database Cache for Resumability
        cached = db.get_prediction(run_id, 0, item.name)
        if cached is not None:
            pred_data, exact_match_score, p_tokens, c_tokens, cost = cached
            logger.info(f"Loaded cached baseline prediction for '{item.name}' (F1: {exact_match_score:.4f})")
            
            # Recreate raw response if output prediction json file exists
            raw_response = ""
            pred_out_path = os.path.join(preds_dir, f"{item.name}_pred.json")
            if os.path.exists(pred_out_path):
                try:
                    with open(pred_out_path, "r") as f:
                        raw_response = json.load(f).get("raw_response", "")
                except Exception:
                    pass
            
            results.append({
                "name": item.name,
                "score": exact_match_score,
                "gold": gold_data,
                "prediction": pred_data,
                "raw_response": raw_response
            })
            total_score += exact_match_score
            total_prompt_tokens += p_tokens
            total_candidate_tokens += c_tokens
            total_cost += cost
            print(f" -> Exact Match Score (F1): {exact_match_score:.4f} (Loaded Cache)")
            continue

        # 6b. Parse PDF
        try:
            resume_text = extract_text_from_pdf(item.pdf_path)
        except Exception as e:
            logger.error(f"Failed to prepare input for {item.name}: {e}")
            continue

        # 6c. Format prompt and determine multimodal status
        if not resume_text or len(resume_text.strip()) < 10:
            logger.info(f"Extracted text for {item.name}.pdf is empty or too short. Using multimodal PDF processing.")
            user_prompt = "Please extract the candidate's details from the attached PDF document."
            pdf_path_param = item.pdf_path
        else:
            logger.info(f"Extracted text successfully for {item.name}.pdf. Using text-based prompt.")
            user_prompt = format_extraction_prompt(resume_text)
            pdf_path_param = None

        # 6d. Call LLM (unpack returns including tokens and cost)
        try:
            raw_response, prompt_tokens, candidate_tokens, cost = client.extract_structured_data(
                prompt=user_prompt,
                system_instruction=seed_prompt,
                response_schema=schema_class,
                pdf_path=pdf_path_param,
                model_name=config.model_name,
                temperature=config.model_temperature,
                max_output_tokens=config.model_max_output_tokens
            )
        except Exception as e:
            logger.error(f"API Call failed for {item.name}: {e}")
            continue

        # 6e. Parse and Validate
        try:
            pred_schema = parse_json_to_schema(raw_response, schema_class)
            pred_data = pred_schema.model_dump()
        except Exception as e:
            logger.error(f"Parsing/Validation failed for {item.name}: {e}")
            pred_data = {}

        # 6f. Evaluate
        exact_match_score = compute_exact_match_score(pred_data, gold_data, schema_dict, client, db)
        total_score += exact_match_score
        
        total_prompt_tokens += prompt_tokens
        total_candidate_tokens += candidate_tokens
        total_cost += cost

        # Save prediction details to SQLite Database
        db.save_prediction(
            run_id=run_id,
            step_index=0,
            file_name=item.name,
            f1_score=exact_match_score,
            prediction_dict=pred_data,
            gold_dict=gold_data,
            prompt_tokens=prompt_tokens,
            candidate_tokens=candidate_tokens,
            cost=cost
        )

        # Save single prediction to file
        pred_out_path = os.path.join(preds_dir, f"{item.name}_pred.json")
        with open(pred_out_path, "w") as f:
            json.dump({
                "name": item.name,
                "score": exact_match_score,
                "gold": gold_data,
                "prediction": pred_data,
                "raw_response": raw_response,
                "prompt_tokens": prompt_tokens,
                "candidate_tokens": candidate_tokens,
                "cost": cost
            }, f, indent=4)

        results.append({
            "name": item.name,
            "score": exact_match_score,
            "gold": gold_data,
            "prediction": pred_data
        })
        print(f" -> Exact Match Score (F1): {exact_match_score:.4f}")

    # Register baseline step 0 in Steps table
    mean_score = total_score / len(results) if results else 0.0
    db.save_step(run_id, 0, seed_prompt, mean_score, "BASELINE")

    # 7. Save Summary Report
    summary_path = os.path.join(run_dir, "summary.json")
    
    # Extract prediction/gold for aggregation
    all_preds_data = [res["prediction"] for res in results if res.get("prediction")]
    all_golds_data = [res["gold"] for res in results if res.get("gold")]
    
    aggregated_metrics = {}
    if all_preds_data and all_golds_data:
        try:
            aggregated_metrics = aggregate_run_metrics(all_preds_data, all_golds_data, schema_dict, client, db)
        except Exception as e:
            logger.error(f"Failed to compute aggregated metrics: {e}")

    summary_info = {
        "timestamp": timestamp,
        "model": config.model_name,
        "system_instruction": seed_prompt,
        "mean_score": mean_score,
        "token_telemetry": {
            "prompt_tokens": total_prompt_tokens,
            "candidate_tokens": total_candidate_tokens,
            "cost_usd": total_cost
        },
        "results": [
            {"name": res["name"], "score": res["score"]}
            for res in results
        ],
        "aggregated_metrics": aggregated_metrics
    }

    with open(summary_path, "w") as f:
        json.dump(summary_info, f, indent=4)

    print("\n" + "="*60)
    print("BASELINE RUN COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {run_dir}")
    print(f"Mean Validation Exact Match Score (F1): {mean_score:.4f}")
    print(f"Total API Cost: ${total_cost:.6f} (Tokens: In={total_prompt_tokens}, Out={total_candidate_tokens})")
    print("-"*60)
    print(f"{'Filename':<25} | {'Score':<10}")
    print("-"*60)
    for res in results:
        print(f"{res['name']:<25} | {res['score']:<10.4f}")
        
    if aggregated_metrics:
        print("\n" + "="*60)
        print("AGGREGATED METRICS ACROSS DOCUMENTS")
        print("="*60)
        print(f"Global Precision: {aggregated_metrics['global']['precision']:.4f}")
        print(f"Global Recall:    {aggregated_metrics['global']['recall']:.4f}")
        print(f"Global F1 Score:  {aggregated_metrics['global']['f1']:.4f}")
        print("-"*60)
        print(f"{'Subtree (Category)':<30} | {'F1 Score':<10}")
        print("-"*60)
        for sub_name, sub_scores in sorted(aggregated_metrics["per_subtree"].items()):
            print(f"{sub_name:<30} | {sub_scores['f1']:<10.4f}")
        print("-"*60)
        print(f"{'Leaf Field (Schema Path)':<50} | {'F1 Score':<10}")
        print("-"*60)
        for leaf_name, leaf_scores in sorted(aggregated_metrics["per_leaf"].items()):
            print(f"{leaf_name:<50} | {leaf_scores['f1']:<10.4f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Prompt Extraction Runner")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    run_baseline(args.config)
