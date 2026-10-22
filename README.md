# Receipt Text → JSON: QLoRA Fine-Tuning of Qwen2.5-3B on CORD-v2

A single Google Colab notebook (`Receipt_Extraction_Fine_Tuning.ipynb`) that fine-tunes a small open LLM to convert **receipt text** into a fixed JSON schema, evaluates it against the un-tuned base model, and exports a quantized GGUF file.

## What this project is (and is not)

- **Is:** a text-to-JSON information-extraction experiment. Input is receipt text, output is structured JSON.
- **Is not:** an end-to-end receipt scanner. The model never sees images. The input text is built from CORD-v2's **ground-truth OCR word annotations**, so the results do not include any OCR errors from a real OCR engine.
- **Is not:** a production system. It is a small-scale experiment (100 test receipts, one training run, one seed).

## Setup

| Item | Value |
|---|---|
| Dataset | `naver-clova-ix/cord-v2` (800 train / 100 validation / 100 test) |
| Base model | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` |
| Method | QLoRA via Unsloth + TRL `SFTTrainer` |
| LoRA | r=16, alpha=32, dropout 0, all attention + MLP projection layers |
| Training | 2 epochs, lr 2e-4, batch 2 × grad-accum 4, cosine schedule, 8-bit AdamW, max seq length 2048, seed 3407 |
| Loss | Computed on the assistant's JSON answer only (`train_on_responses_only`) |
| Hardware | Free Colab T4 GPU (training took ~1,400 s) |

## Pipeline

1. **Data preparation.** For each CORD receipt, the text is rebuilt from `valid_line` word boxes: words are grouped by `row_id` and sorted left-to-right by x-coordinate. The target JSON is built from `gt_parse` with this schema:
   - `items`: list of `{name, quantity, price}`
   - `subtotal`, `tax`, `total`, `cash`, `change` (all optional strings)

   Values are kept as written in CORD (e.g. quantity `"1 x"`, price `"40,000."`). Targets are validated against a Pydantic model. Longest training example is 1,261 tokens, so no training examples were dropped by the 2048 limit.
2. **Baseline.** The 4-bit base model is run zero-shot with a system prompt containing the JSON schema.
3. **Fine-tuning.** LoRA adapter trained on the 800 training receipts. Validation loss was logged every 50 steps; no checkpoint selection was done (`save_strategy="no"`). Adapter saved.
4. **Evaluation** on the 100-receipt test set with greedy decoding (see metrics below).
5. **Analysis:** error analysis, leakage check, small robustness test.
6. **Export** to `q4_k_m` GGUF and re-evaluation of that file with `llama-cpp-python`, with and without grammar-constrained JSON decoding.

## Metrics

- **JSON valid / schema valid:** output parses as JSON / validates against the Pydantic schema.
- **Field F1:** micro-F1 over the five scalar fields (`subtotal`, `tax`, `total`, `cash`, `change`).
- **Item F1:** micro-F1 over `(name, quantity, price)` tuples, compared as multisets.
- **Exact match:** all five scalar fields and the full item multiset match.
- All comparisons are **normalized**: lowercased, with everything except `a–z` and `0–9` removed. So `"40,000."` and `"40000"` count as equal, and the metric does not check formatting.

## Results (test split, n = 100)

| Model | JSON valid | Schema valid | Field F1 | Item F1 | Exact match |
|---|---|---|---|---|---|
| Base, zero-shot (4-bit) | 100% | 95% | 0.753 | 0.394 | 8% (95% CI 3–14) |
| Fine-tuned (4-bit + LoRA) | 100% | 100% | 0.983 | 0.877 | 71% (95% CI 62–80) |
| GGUF `q4_k_m`, no grammar | 100% | 100% | 0.968 | 0.857 | 65% |
| GGUF `q4_k_m`, + JSON grammar | 100% | 100% | 0.968 | 0.857 | 65% |

Confidence intervals are bootstrap (2,000 resamples) over the 100 test receipts. With n = 100 they are wide.

Speed on the T4 (GGUF, one receipt at a time): about **1.41 s/receipt without grammar** and **5.35 s/receipt with grammar**. Grammar-constrained decoding changed none of the metrics here, only latency.

### Leakage check

Test receipts were compared with training receipts using token-set Jaccard overlap. 25 of 100 test receipts had overlap above 0.8 with a training receipt (median overlap 0.44). After removing those, exact match is **69.3%** for the fine-tuned model (n = 75, 95% CI 58.7–78.7) and 6.7% for the base model. Exact match was 82% on test receipts with overlap above 0.5 (n = 40) versus 63% on the rest (n = 60). Near-duplicates therefore help somewhat, but the main gain over the baseline remains.

### Robustness (one small stress test)

Every keyword (TOTAL, CASH, CHANGE, etc.) in the test receipts was corrupted with 1 or 2 random character edits, using a single noise draw:

| Condition | Field F1 | Item F1 | Exact match | `cash` F1 |
|---|---|---|---|---|
| Clean | 0.983 | 0.877 | 71% | 0.969 |
| 1-edit keyword noise | 0.872 | 0.874 | 48% | 0.699 |
| 2-edit keyword noise | 0.807 | 0.855 | 36% | 0.494 |

Performance drops noticeably when the summary keywords are corrupted, especially for `cash`. This is a synthetic test and does not represent real OCR noise.

### Error analysis

The notebook prints the 29 non-exact test predictions and categorizes them (wrong item count, item field mismatch, scalar mismatch). A manual audit covered **14 of the 29 failures**, which were chosen because they looked suspicious, not randomly. The notebook does not give a complete breakdown of all failures, so no failure-rate claims beyond the metrics above are made here.

### Negative result

A rule-based post-filter that removes summary lines (total, tax, cash…) wrongly predicted as items was tried on the **validation** set. According to the notebook it had no effect and is not used in the final pipeline.

## Limitations

- Input is clean ground-truth OCR text from CORD-v2, not raw OCR output or images.
- CORD-v2 is dominated by Indonesian receipts, so results may not transfer to other layouts, languages or currencies. This was not tested.
- Test set is only 100 receipts, one training run, one seed. Differences of a few points should not be treated as meaningful.
- The extraction schema is limited to line items plus five totals fields. Fields such as merchant, date or payment method are not extracted.
- Normalized matching ignores punctuation and case, so formatting errors are not penalized.
- Quantization cost some accuracy: exact match fell from 71% to 65% (field F1 0.983 → 0.968).
- The Starbucks-style demo receipts in the notebook are two hand-written examples, not an evaluation.
- Runtimes are from a free Colab T4 and are not benchmarks of other hardware.

## Files produced (saved to Google Drive under `receipt_extraction_llm/`)

- `config.json`: hyperparameters
- `lora_adapter/`: LoRA adapter and tokenizer
- `system_prompt.txt`: the exact training prompt (inference must use the same prompt)
- `results.json`: all metrics
- `loss_curve.png`, `results.png`: training curve and results chart
- `gguf_model/model.gguf`: `q4_k_m` GGUF (1.93 GB)

## How to run

1. Open the notebook in Colab and set the runtime to **T4 GPU**.
2. Run the cells in order. Section 1 mounts Google Drive.
3. GGUF export needs about 14 GB of working disk and takes roughly 15–20 minutes. Building `llama-cpp-python` with CUDA for section 14 can take another 10–20 minutes.
4. After a runtime reset, do **not** use "Run all". Follow the resume block at the start of section 14 instead.

If you hit out-of-memory errors, the notebook suggests switching to `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`. That variant was not run here.
