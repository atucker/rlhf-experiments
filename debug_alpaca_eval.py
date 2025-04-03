from alpaca_eval import evaluate
from alpaca_eval.metrics.glm_winrate import get_length_controlled_winrate
# Run the evaluation
evaluate(
    model_outputs="src/dips/ultrafeedback/alpaca_eval_results.json",
    fn_metric="get_length_controlled_winrate"  # Explicitly set the metric function
) 