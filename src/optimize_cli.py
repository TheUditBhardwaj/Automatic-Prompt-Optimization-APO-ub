import argparse
import logging
from config import Config
from dataset import load_dataset, split_dataset
from llm_wrapper import GeminiClient
from optimizer import GreedyOptimizer

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def main(config_path: str, resume_run_id: str = None):
    """Orchestrator for automated prompt optimization."""
    # 1. Load config
    config = Config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # 2. Load dataset
    all_items = load_dataset(config.raw_dir, config.gold_dir)
    if not all_items:
        raise ValueError(f"No matched items found in raw_dir: {config.raw_dir}")

    # 3. Split dataset
    train_items, val_items, test_items = split_dataset(
        items=all_items,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed
    )

    if not val_items:
        logger.warning("Validation split is empty! Running optimization on entire dataset.")
        eval_items = all_items
    else:
        eval_items = val_items

    # 4. Initialize Gemini client
    try:
        client = GeminiClient()
    except Exception as e:
        logger.error(f"Failed to initialize Gemini Client: {e}")
        return

    # 5. Run Optimizer (with optional resume_run_id)
    optimizer = GreedyOptimizer(
        config=config,
        client=client,
        val_items=eval_items,
        test_items=test_items,
        run_id=resume_run_id
    )
    
    optimizer.run_optimization()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Prompt Optimization Loop")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Run ID of the optimization run to resume (e.g. optimize_20260523_171110)"
    )
    args = parser.parse_args()
    main(args.config, args.resume)
