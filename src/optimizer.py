import os
import json
import logging
import difflib
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid GUI window popup issues
import matplotlib.pyplot as plt

from config import Config
from schema import get_schema_class, load_schema_dict
from dataset import DatasetItem
from pdf_loader import extract_text_from_pdf
from prompts import DEFAULT_SYSTEM_INSTRUCTION, get_seed_prompt, format_extraction_prompt
from llm_wrapper import GeminiClient
from parser import parse_json_to_schema
from evaluator import evaluate_predictions, aggregate_run_metrics
from mutator import mutate_prompt
from database import APODatabase

logger = logging.getLogger(__name__)

class GreedyOptimizer:
    """Orchestrates LLM-guided greedy prompt optimization based on validation F1 scores with SQLite persistence."""

    def __init__(
        self,
        config: Config,
        client: GeminiClient,
        val_items: List[DatasetItem],
        test_items: Optional[List[DatasetItem]] = None,
        run_id: Optional[str] = None
    ):
        self.config = config
        self.client = client
        self.val_items = val_items
        self.test_items = test_items or []
        
        # Load schema class and schema dict dynamically
        self.schema_class = get_schema_class(config.schema_type)
        self.schema_dict = load_schema_dict(config.schema_type)
        
        # Initialize SQLite database
        self.db = APODatabase()
        
        # If resuming, use the existing run_id, otherwise generate a new one
        self.is_resume = run_id is not None
        self.run_id = run_id or f"optimize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.timestamp = self.run_id.split("_")[-1] if "_" in self.run_id else self.run_id
        
        # Prepare optimization run folder
        self.run_dir = os.path.join(config.output_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Save run config
        with open(os.path.join(self.run_dir, "config_run.yaml"), "w") as f:
            json.dump(config.data, f, indent=4)

        # Resolve the seed prompt for this schema
        self.seed_prompt = get_seed_prompt(config.schema_type, config.seed_prompt)

    def _evaluate_prompt(
        self, 
        system_instruction: str, 
        step_index: int, 
        eval_items: Optional[List[DatasetItem]] = None
    ) -> Tuple[float, List[Dict[str, Any]], int, int, float]:
        """Runs extraction and scores predictions over validation or test items, checking SQLite cache first."""
        items_to_eval = eval_items if eval_items is not None else self.val_items
        results = []
        total_f1 = 0.0
        step_prompt_tokens = 0
        step_candidate_tokens = 0
        step_cost = 0.0

        for item in items_to_eval:
            gold_data = item.load_gold_json()
            
            # 1. Check SQLite cache first for resumability
            cached = self.db.get_prediction(self.run_id, step_index, item.name)
            if cached is not None:
                pred_data, f1, prompt_tokens, candidate_tokens, cost = cached
                logger.info(f"Loaded cached prediction for '{item.name}' at step {step_index} (F1: {f1:.4f})")
                
                # Fetch raw response from cached file if it exists, otherwise empty string
                raw_response = ""
                pred_out_path = os.path.join(self.run_dir, f"step_{step_index}", f"{item.name}_pred.json")
                if os.path.exists(pred_out_path):
                    try:
                        with open(pred_out_path, "r") as f:
                            raw_response = json.load(f).get("raw_response", "")
                    except Exception:
                        pass
                
                results.append({
                    "name": item.name,
                    "score": f1,
                    "gold": gold_data,
                    "prediction": pred_data,
                    "raw_response": raw_response,
                    "prompt_tokens": prompt_tokens,
                    "candidate_tokens": candidate_tokens,
                    "cost": cost
                })
                total_f1 += f1
                step_prompt_tokens += prompt_tokens
                step_candidate_tokens += candidate_tokens
                step_cost += cost
                continue

            # 2. No cache found: Proceed to extract
            try:
                resume_text = extract_text_from_pdf(item.pdf_path)
            except Exception as e:
                logger.error(f"Failed to read PDF for {item.name}: {e}")
                continue
            
            # Formulate prompt
            if not resume_text or len(resume_text.strip()) < 10:
                user_prompt = "Please extract details from the attached PDF document."
                pdf_path_param = item.pdf_path
            else:
                user_prompt = format_extraction_prompt(resume_text)
                pdf_path_param = None

            # Call LLM wrapper (returns text, prompt_tokens, candidate_tokens, cost)
            try:
                raw_response, prompt_tokens, candidate_tokens, cost = self.client.extract_structured_data(
                    prompt=user_prompt,
                    system_instruction=system_instruction,
                    response_schema=self.schema_class,
                    pdf_path=pdf_path_param,
                    model_name=self.config.model_name,
                    temperature=self.config.model_temperature,
                    max_output_tokens=self.config.model_max_output_tokens
                )
            except Exception as e:
                logger.error(f"LLM API Call failed for {item.name}: {e}")
                continue

            # Parse
            try:
                pred_schema = parse_json_to_schema(raw_response, self.schema_class)
                pred_data = pred_schema.model_dump()
            except Exception as e:
                logger.error(f"Parse/Validation failed for {item.name}: {e}")
                pred_data = {}

            # Evaluate
            score_metrics = evaluate_predictions(pred_data, gold_data, self.schema_dict, self.client, self.db)
            f1 = score_metrics["f1"]
            total_f1 += f1

            # Save to SQLite database
            self.db.save_prediction(
                run_id=self.run_id,
                step_index=step_index,
                file_name=item.name,
                f1_score=f1,
                prediction_dict=pred_data,
                gold_dict=gold_data,
                prompt_tokens=prompt_tokens,
                candidate_tokens=candidate_tokens,
                cost=cost
            )

            results.append({
                "name": item.name,
                "score": f1,
                "gold": gold_data,
                "prediction": pred_data,
                "raw_response": raw_response,
                "prompt_tokens": prompt_tokens,
                "candidate_tokens": candidate_tokens,
                "cost": cost,
                "metrics": score_metrics
            })
            step_prompt_tokens += prompt_tokens
            step_candidate_tokens += candidate_tokens
            step_cost += cost
            
        mean_f1 = total_f1 / len(results) if results else 0.0
        return mean_f1, results, step_prompt_tokens, step_candidate_tokens, step_cost

    def run_optimization(self) -> Tuple[str, float]:
        """Runs the main greedy prompt optimization loop with persistence and resumability."""
        print("\n" + "="*60)
        if self.is_resume:
            print(f"RESUMING AUTOMATED PROMPT OPTIMIZATION Run ID: {self.run_id}")
        else:
            print(f"STARTING AUTOMATED PROMPT OPTIMIZATION Run ID: {self.run_id}")
        print(f"Iterations: {self.config.opt_iterations}")
        print(f"Validation dataset size: {len(self.val_items)} items")
        print("="*60)

        # 1. Save Run metadata to SQLite
        self.db.save_run(
            run_id=self.run_id,
            timestamp=self.timestamp,
            model_name=self.config.model_name,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            seed=self.config.seed,
            config_dict=self.config.data
        )

        total_run_cost = 0.0
        total_prompt_tokens = 0
        total_candidate_tokens = 0

        # Step 1: Establish baseline with seed prompt (Step 0)
        print("\n[Step 0] Evaluating seed prompt (Baseline)...")
        
        cached_step_0 = self.db.get_step(self.run_id, 0)
        if cached_step_0:
            print(" -> Loaded step 0 details from database cache.")
            best_prompt = cached_step_0["prompt"]
            best_score = cached_step_0["mean_f1"]
            results = self.db.get_run_predictions(self.run_id, 0)
        else:
            best_prompt = self.seed_prompt
            best_score, results, p_tokens, c_tokens, cost = self._evaluate_prompt(best_prompt, 0)
            
            total_prompt_tokens += p_tokens
            total_candidate_tokens += c_tokens
            total_run_cost += cost
            
            # Save step to SQLite
            self.db.save_step(self.run_id, 0, best_prompt, best_score, "BASELINE")
            
        print(f" -> Baseline Mean F1: {best_score:.4f}")
        self._save_step_predictions(0, results)

        current_prompt = best_prompt
        current_results = results

        # Step 2: Loop iterations
        for i in range(1, self.config.opt_iterations + 1):
            print(f"\n[Step {i}/{self.config.opt_iterations}] Mutating prompt...")
            
            cached_step = self.db.get_step(self.run_id, i)
            if cached_step:
                print(f" -> Loaded step {i} details from database cache.")
                score = cached_step["mean_f1"]
                mutated_prompt = cached_step["prompt"]
                status = cached_step["status"]
                mutated_results = self.db.get_run_predictions(self.run_id, i)
                
                # Update current best values accordingly if accepted
                if status == "ACCEPT":
                    best_score = score
                    best_prompt = mutated_prompt
                    current_prompt = mutated_prompt
                    current_results = mutated_results
                print(f" -> Cached Step Mean F1: {score:.4f} (Status: {status})")
                continue

            # No cache for this step: run mutation and evaluation
            try:
                # Compile mutation based on current predictions
                mutated_prompt = mutate_prompt(
                    client=self.client,
                    current_instruction=current_prompt,
                    results=current_results,
                    model_name=self.config.opt_mutation_model,
                    temperature=self.config.opt_mutation_temperature
                )
            except Exception as e:
                logger.error(f"Mutation failed at iteration {i}: {e}. Skipping iteration.")
                continue

            print(f"[Step {i}/{self.config.opt_iterations}] Evaluating mutated prompt on validation set...")
            mutated_score, mutated_results, p_tokens, c_tokens, cost = self._evaluate_prompt(mutated_prompt, i)
            
            total_prompt_tokens += p_tokens
            total_candidate_tokens += c_tokens
            total_run_cost += cost
            
            print(f" -> Mutated Prompt Mean F1: {mutated_score:.4f} (Previous Best: {best_score:.4f})")

            # Greedy acceptance check
            if mutated_score > best_score:
                print(" >>> ACCEPTED: Improved validation score!")
                best_score = mutated_score
                best_prompt = mutated_prompt
                current_prompt = mutated_prompt
                current_results = mutated_results
                status = "ACCEPT"
            else:
                print(" >>> REJECTED: No score improvement.")
                status = "REJECT"

            # Save predictions and commit step to SQLite
            self._save_step_predictions(i, mutated_results)
            self.db.save_step(self.run_id, i, mutated_prompt, mutated_score, status)

        # Step 3: Run final evaluation on the test split (both seed and final prompt)
        test_summary = {}
        if self.test_items:
            print("\n" + "="*60)
            print("RUNNING FINAL HELD-OUT TEST EVALUATION")
            print("="*60)
            
            # Seed prompt test evaluation
            print("Evaluating seed prompt (Baseline) on test split...")
            seed_test_f1, seed_test_results, seed_p_tokens, seed_c_tokens, seed_cost = self._evaluate_prompt(
                self.seed_prompt, 
                step_index=-1,  # use step_index -1 for seed test
                eval_items=self.test_items
            )
            total_prompt_tokens += seed_p_tokens
            total_candidate_tokens += seed_c_tokens
            total_run_cost += seed_cost
            
            # Final prompt test evaluation
            print("Evaluating final optimized prompt on test split...")
            final_test_f1, final_test_results, final_p_tokens, final_c_tokens, final_cost = self._evaluate_prompt(
                best_prompt, 
                step_index=-2,  # use step_index -2 for final test
                eval_items=self.test_items
            )
            total_prompt_tokens += final_p_tokens
            total_candidate_tokens += final_c_tokens
            total_run_cost += final_cost
            
            # Compute leaf/subtree aggregates for seed vs. final
            seed_preds = [res["prediction"] for res in seed_test_results]
            seed_golds = [res["gold"] for res in seed_test_results]
            seed_agg = aggregate_run_metrics(seed_preds, seed_golds, self.schema_dict, self.client, self.db)
            
            final_preds = [res["prediction"] for res in final_test_results]
            final_golds = [res["gold"] for res in final_test_results]
            final_agg = aggregate_run_metrics(final_preds, final_golds, self.schema_dict, self.client, self.db)
            
            test_summary = {
                "seed_test_f1": seed_test_f1,
                "final_test_f1": final_test_f1,
                "seed_aggregation": seed_agg,
                "final_aggregation": final_agg
            }
            
            # Print beautiful comparison
            print("\n" + "="*60)
            print("HELD-OUT TEST SPLIT COMPARATIVE REPORT")
            print("="*60)
            print(f"Seed Prompt Test F1:          {seed_test_f1:.4f}")
            print(f"Final Optimized Test F1:      {final_test_f1:.4f}")
            print(f"Relative Improvement:         {((final_test_f1 - seed_test_f1)/seed_test_f1 * 100 if seed_test_f1 > 0 else 0):+.1f}%")
            print("-"*60)
            print(f"{'Subtree (Category)':<30} | {'Seed F1':<10} | {'Final F1':<10} | {'Change':<8}")
            print("-"*60)
            for sub_name in sorted(set(seed_agg["per_subtree"].keys()).union(final_agg["per_subtree"].keys())):
                s_f1 = seed_agg["per_subtree"].get(sub_name, {}).get("f1", 0.0)
                f_f1 = final_agg["per_subtree"].get(sub_name, {}).get("f1", 0.0)
                print(f"{sub_name:<30} | {s_f1:<10.4f} | {f_f1:<10.4f} | {f_f1 - s_f1:+.4f}")
            print("="*60 + "\n")

        # Step 4: Optimization complete: generate plots, prompt diffs, final summaries
        self._generate_plots()
        self._generate_prompt_diffs(self.seed_prompt, best_prompt)
        self._save_final_outputs(best_prompt, best_score, total_prompt_tokens, total_candidate_tokens, total_run_cost, test_summary)

        print("\n" + "="*60)
        print("PROMPT OPTIMIZATION COMPLETED")
        print(f"Trajectory score curve saved to: {self.run_dir}/score_curve.png")
        print(f"Prompt diff text saved to: {self.run_dir}/prompt_diff.txt")
        print(f"Baseline Score: {self.db.get_step(self.run_id, 0)['mean_f1']:.4f}")
        print(f"Final Optimized Score: {best_score:.4f}")
        print(f"Total API Cost for Run: ${total_run_cost:.6f} (Tokens: In={total_prompt_tokens}, Out={total_candidate_tokens})")
        print("="*60 + "\n")
        
        return best_prompt, best_score

    def _save_step_predictions(self, iteration: int, results: List[Dict[str, Any]]):
        """Saves prediction files to a subfolder for reference/debugging."""
        step_dir = os.path.join(self.run_dir, f"step_{iteration}")
        os.makedirs(step_dir, exist_ok=True)
        for res in results:
            item_path = os.path.join(step_dir, f"{res['name']}_pred.json")
            with open(item_path, "w") as f:
                json.dump(res, f, indent=4)

    def _generate_plots(self):
        """Generates trajectory validation curves using matplotlib."""
        steps = self.db.get_run_steps(self.run_id)
        if not steps:
            return
            
        x = [s["step_index"] for s in steps]
        y = [s["mean_f1"] for s in steps]
        
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, marker='o', linewidth=2.5, color='#4F46E5', label='Validation F1')
        
        # Annotate status tags (Baseline, Accept, Reject)
        for s in steps:
            status = s["status"]
            idx = s["step_index"]
            val = s["mean_f1"]
            if status == "ACCEPT":
                plt.annotate("Accept", (idx, val), textcoords="offset points", xytext=(0,10), ha='center', color='green', fontweight='bold')
            elif status == "REJECT":
                plt.annotate("Reject", (idx, val), textcoords="offset points", xytext=(0,-15), ha='center', color='red')
            elif status == "BASELINE":
                plt.annotate("Baseline", (idx, val), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
                
        plt.title('Prompt Optimization F1 Score Trajectory', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Step', fontsize=12)
        plt.ylabel('Mean F1 Score', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(x)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='lower right')
        
        plot_path = os.path.join(self.run_dir, "score_curve.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"Saved score trajectory curve to {plot_path}")

    def _generate_prompt_diffs(self, seed_prompt: str, final_prompt: str):
        """Computes line-by-line diffs between the seed prompt and optimized prompt."""
        diff = list(difflib.unified_diff(
            seed_prompt.splitlines(keepends=True),
            final_prompt.splitlines(keepends=True),
            fromfile='Seed Prompt',
            tofile='Optimized Prompt',
            n=2
        ))
        diff_text = "".join(diff)
        
        diff_path = os.path.join(self.run_dir, "prompt_diff.txt")
        with open(diff_path, "w") as f:
            f.write(diff_text)
            
        print("\n" + "-"*40)
        print("PROMPT DIFF (Baseline -> Final Optimized)")
        print("-"*40)
        if diff_text:
            print(diff_text)
        else:
            print("No changes made to prompt.")
        print("-"*40 + "\n")

    def _save_final_outputs(
        self, 
        final_prompt: str, 
        final_score: float, 
        p_tokens: int, 
        c_tokens: int, 
        cost: float,
        test_summary: Dict[str, Any]
    ):
        """Saves final optimized prompt file and summary stats."""
        # 1. Save final prompt
        final_prompt_path = os.path.join(self.run_dir, "final_prompt.txt")
        with open(final_prompt_path, "w") as f:
            f.write(final_prompt)
            
        # 2. Save final metadata summary
        summary_path = os.path.join(self.run_dir, "summary.json")
        steps = self.db.get_run_steps(self.run_id)
        
        summary = {
            "run_id": self.run_id,
            "baseline_score": steps[0]["mean_f1"] if steps else 0.0,
            "optimized_score": final_score,
            "total_iterations": self.config.opt_iterations,
            "token_telemetry": {
                "prompt_tokens": p_tokens,
                "candidate_tokens": c_tokens,
                "cost_usd": cost
            },
            "trajectory": [
                {
                    "step": t["step_index"],
                    "status": t["status"],
                    "score": t["mean_f1"]
                }
                for t in steps
            ],
            "held_out_test_results": test_summary
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
