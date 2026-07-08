[Introducing Unsloth Studio: a new web UI for local AI\
\
🦥](https://unsloth.ai/docs/new/studio)

For the complete documentation index, see [llms.txt](https://unsloth.ai/docs/llms.txt). This page is also available as [Markdown](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs.md).

We're excited to introduce [Unsloth](https://github.com/unslothai/unsloth) Dynamic v2.0 quantization - a major upgrade to our previous quants. This new method outperforms leading quantization methods and sets new benchmarks for [Aider Polyglot](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot), 5-shot MMLU and KL Divergence.

This means you can now run + fine-tune [quantized LLMs](https://unsloth.ai/docs/models/tutorials) while preserving as much accuracy as possible! You can run the 2.0 GGUFs on most inference engines like llama.cpp, [Unsloth Studio](https://unsloth.ai/docs/new/studio) etc.

**Apr 20, 2026 Update:** See our new GGUF Benchmarks for [Qwen3.6](https://unsloth.ai/docs/models/qwen3.6#unsloth-gguf-benchmarks) and [Gemma 4](https://unsloth.ai/docs/models/gemma-4#unsloth-gguf-benchmarks).

[Feb 27, 2026 Update:](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) **Qwen3.5** is out and we fixed some tool-calling chat template issues and benchmarked every GGUF on perplexity & KL Divergence. [See benchmarks!](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)

The **key advantage** of using the [Unsloth package](https://github.com/unslothai/unsloth) and quants is our active role in fixing bugs in major models. We've collaborated directly with teams behind [Qwen3](https://www.reddit.com/r/LocalLLaMA/comments/1kaodxu/qwen3_unsloth_dynamic_ggufs_128k_context_bug_fixes/), [Meta (Llama 4)](https://github.com/ggml-org/llama.cpp/pull/12889), [Mistral (Devstral)](https://app.gitbook.com/o/HpyELzcNe0topgVLGCZY/s/xhOjnexMCB3dmuQFQ2Zq/~/changes/618/basics/tutorials-how-to-fine-tune-and-run-llms/devstral-how-to-run-and-fine-tune), [Google (Gemma 1–3)](https://news.ycombinator.com/item?id=39671146) and [Microsoft (Phi-3/4)](https://simonwillison.net/2025/Jan/11/phi-4-bug-fixes), contributing fixes that increase accuracy.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FtRVN97QDzO0Pq7SscC7x%252Fgemma%2520426b%2520bench.png%3Falt%3Dmedia%26token%3D80b4da76-efe9-4554-8e31-cca6494d456c&width=768&dpr=3&quality=100&sign=b32dccb3&sv=2)

Gemma 4 26B A4B Benchmarks (lower is better)

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FHq98A18pHA2ePwlInrFG%252Fqwen36_mean_q6k_corrected_arrow_pareto_fixed.png%3Falt%3Dmedia%26token%3Da5190c8a-4d04-4d4d-be94-dd15214e6687&width=768&dpr=3&quality=100&sign=d0938e2f&sv=2)

Qwen3.6 Benchmarks (lower is better)

Unsloth Dynamic GGUFs can now be run in [Unsloth Studio](https://unsloth.ai/docs/new/studio) ✨

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FsERzR65n4jc1UtuSHozZ%252Fwedefrwfwe.gif%3Falt%3Dmedia%26token%3De303ed9e-0d90-456d-8d57-874a06803903&width=300&dpr=3&quality=100&sign=711edce8&sv=2)

[Sept 10, 2025 update:](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) You asked for tougher benchmarks, so here's Aider Polyglot results! Our Dynamic 3-bit DeepSeek V3.1 GGUF scores **75.6%**, surpassing many full-precision SOTA LLMs. [Read more.](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)

![DeepSeek-V3.2 Thinking Aider Benchmarks](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-a114143bdd47add988182aabf9313ab40be38d7d%252Faider%2520thinking.png%3Falt%3Dmedia&width=300&dpr=3&quality=100&sign=6d6a39cd&sv=2)![Llama 4 5-shot MMLU Benchmarks](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-b085c16c7f8351308229f1341846cbf1a2617d0a%252Faider%2520non.png%3Falt%3Dmedia&width=300&dpr=3&quality=100&sign=37ebacc0&sv=2)

You can also view real-world use-case benchmarks conducted by Benjamin Marie for LiveCodeBench v6, MMLU Pro etc.:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FhfO2gsbz2lWrZXg3ojyE%252FHCGBTzgboAASv_A.png%3Falt%3Dmedia%26token%3D7d6334ca-4f3c-4946-aacd-d55527375fce&width=768&dpr=3&quality=100&sign=d36871c9&sv=2)

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Ftbfnqq8ppzwFbeqPhnw0%252FHAfMRrrXQAALkQb.png%3Falt%3Dmedia%26token%3D9730d4e1-3d4a-4ae6-92bf-32aa6724ab86&width=768&dpr=3&quality=100&sign=d24449f1&sv=2)

You can see how Unsloth's GGUFs performs better than the non-Unsloth quants despite being ~8GB smaller.

Detailed analysis of our benchmarks and evaluation further below.

### [Direct link to heading](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs#whats-new-in-dynamic-v2.0)    💡 What's New in Dynamic v2.0?

- **Revamped Layer Selection for GGUFs + safetensors:** Unsloth Dynamic 2.0 now selectively quantizes layers much more intelligently and extensively. Rather than modifying only select layers, we now dynamically adjust the quantization type of every possible layer, and the combinations will differ for each layer and model.

- Current selected and all future GGUF uploads will utilize Dynamic 2.0 and our new calibration dataset. The dataset contains more than >1.5M **tokens** (depending on model) and comprise of high-quality, hand-curated and cleaned data - to greatly enhance conversational chat performance.

- Previously, our Dynamic quantization (DeepSeek-R1 1.58-bit GGUF) was effective only for MoE architectures. **Dynamic 2.0 quantization now works on all models (including MOEs & non-MoEs)**.

- **Model-Specific Quants:** Each model now uses a custom-tailored quantization scheme. E.g. the layers quantized in Gemma 3 differ significantly from those in Llama 4.

- To maximize efficiency, especially on Apple Silicon and ARM devices, we now also add Q4_NL, Q5.1, Q5.0, Q4.1, and Q4.0 formats.


To ensure accurate benchmarking, we built an internal evaluation framework to match official reported 5-shot MMLU scores of Llama 4 and Gemma 3. This allowed apples-to-apples comparisons between full-precision vs. Dynamic v2.0, **QAT** and standard **imatrix** GGUF quants.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-fd0a92a2bea8efa37b71946ea934a22f00589f40%252Fkldivergence%2520graph.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=6d6a5ae7&sv=2)

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-76662317725a3b76fb1e5e33b586c86e712bee6f%252F5shotmmlu.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=82bd53c&sv=2)

All future GGUF uploads will utilize Unsloth Dynamic 2.0, and our Dynamic 4-bit safe tensor quants will also benefit from this in the future.

## [Direct link to heading](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs#why-kl-divergence)    📊 Why KL Divergence?

[Accuracy is Not All You Need](https://arxiv.org/pdf/2407.09141) showcases how pruning layers, even by selecting unnecessary ones still yields vast differences in terms of "flips". A "flip" is defined as answers changing from incorrect to correct or vice versa. The paper shows how MMLU might not decrease as we prune layers or do quantization,but that's because some incorrect answers might have "flipped" to become correct. Our goal is to match the original model, so measuring "flips" is a good metric.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-5a97c101b0df31fb49df20ce4241930897098cf8%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=fc3b1d30&sv=2)

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-e4a60354ad8613b6f2361f63fa82c552e00fdda9%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=aa571ef6&sv=2)

**KL Divergence** should be **one of the gold standards for reporting quantization errors** as per the research paper "Accuracy is Not All You Need". **Using perplexity is incorrect** since output token values can cancel out, so we must use KLD or harder benchmarks like [Aider](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot).

The paper also shows that interestingly KL Divergence is highly correlated with flips, and so our goal is to reduce the mean KL Divergence whilst increasing the disk space of the quantization as less as possible.

## [Direct link to heading](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs#calibration-dataset-overfitting)    ⚖️ Calibration Dataset Overfitting

Most frameworks report perplexity and KL Divergence using a test set of Wikipedia articles. However, we noticed using the calibration dataset which is also Wikipedia related causes quants to overfit, and attain lower perplexity scores. We utilize [Calibration_v3](https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8) and [Calibration_v5](https://gist.github.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/) datasets for fair testing which includes some wikitext data amongst other data. **Also instruct models have unique chat templates, and using text only calibration datasets is not effective for instruct models** (base models yes). In fact most imatrix GGUFs are typically calibrated with these issues. As a result, they naturally perform better on KL Divergence benchmarks that also use Wikipedia data, since the model is essentially optimized for that domain.

To ensure a fair and controlled evaluation, we do not to use our own calibration dataset (which is optimized for chat performance) when benchmarking KL Divergence. Instead, we conducted tests using the same standard Wikipedia datasets, allowing us to directly compare the performance of our Dynamic 2.0 method against the baseline imatrix approach.

## [Direct link to heading](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs#mmlu-replication-adventure)    🔢 MMLU Replication Adventure

- Replicating MMLU 5 shot was nightmarish. We **could not** replicate MMLU results for many models including Llama 3.1 (8B) Instruct, Gemma 3 (12B) and others due to **subtle implementation issues**. Llama 3.1 (8B) for example should be getting ~68.2%, whilst using incorrect implementations can attain **35% accuracy.**


![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-cc2b4b2bc512b3c9bc065250930259b9b9a9fce0%252FMMLU%2520differences.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=510e99d7&sv=2)

MMLU implementation issues

- Llama 3.1 (8B) Instruct has a MMLU 5 shot accuracy of 67.8% using a naive MMLU implementation. We find however Llama **tokenizes "A" and "_A" (A with a space in front) as different token ids**. If we consider both spaced and non spaced tokens, we get 68.2% (+0.4%)

- Interestingly Llama 3 as per Eleuther AI's [LLM Harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/llama3/instruct/mmlu/_continuation_template_yaml) also appends **"The best answer is"** to the question, following Llama 3's original MMLU benchmarks.

- There are many other subtle issues, and so to benchmark everything in a controlled environment, we designed our own MMLU implementation from scratch by investigating [github.com/hendrycks/test](https://github.com/hendrycks/test) directly, and verified our results across multiple models and comparing to reported numbers.


## [Direct link to heading](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs#gemma-3-qat-replication-benchmarks)    ✨ Gemma 3 QAT Replication, Benchmarks

The Gemma team released two QAT (quantization aware training) versions of Gemma 3:

1. Q4_0 GGUF - Quantizes all layers to Q4_0 via the formula `w = q * block_scale` with each block having 32 weights. See [llama.cpp wiki](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)for more details.

2. int4 version - presumably [TorchAO int4 style](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md)?


We benchmarked all Q4_0 GGUF versions, and did extensive experiments on the 12B model. We see the **12B Q4_0 QAT model gets 67.07%** whilst the full bfloat16 12B version gets 67.15% on 5 shot MMLU. That's very impressive! The 27B model is mostly nearly there!

Metric

1B

4B

12B

27B

MMLU 5 shot

26.12%

55.13%

**67.07% (67.15% BF16)**

**70.64% (71.5% BF16)**

Disk Space

0.93GB

2.94GB

**7.52GB**

16.05GB

**Efficiency***

1.20

10.26

**5.59**

2.84

We designed a new **Efficiency metric** which calculates the usefulness of the model whilst also taking into account its disk size and MMLU 5 shot score:

Efficiency=MMLU 5 shot score−25Disk Space GB\text{Efficiency} = \frac{\text{MMLU 5 shot score} - 25}{\text{Disk Space GB}}Efficiency=Disk Space GBMMLU 5 shot score−25​

We have to **minus 25** since MMLU has 4 multiple choices - A, B, C or D. Assume we make a model that simply randomly chooses answers - it'll get 25% accuracy, and have a disk space of a few bytes. But clearly this is not a useful model.

On KL Divergence vs the base model, below is a table showcasing the improvements. Reminder the closer the KL Divergence is to 0, the better (ie 0 means identical to the full precision model)

Quant

Baseline KLD

GB

New KLD

GB

IQ1_S

1.035688

5.83

0.972932

6.06

IQ1_M

0.832252

6.33

0.800049

6.51

IQ2_XXS

0.535764

7.16

0.521039

7.31

IQ2_M

0.26554

8.84

0.258192

8.96

Q2_K_XL

0.229671

9.78

0.220937

9.95

Q3_K_XL

0.087845

12.51

0.080617

12.76

Q4_K_XL

0.024916

15.41

0.023701

15.64

If we plot the ratio of the disk space increase and the KL Divergence ratio change, we can see a much clearer benefit! Our dynamic 2bit Q2_K_XL reduces KLD quite a bit (around 7.5%).

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-5b352d0449e723556e6e871396c2ee78ae8ec3dc%252Fchart%282%29.svg%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=7edc9709&sv=2)

Truncated table of results for MMLU for Gemma 3 (27B). See below.

1. **Our dynamic 4bit version is 2GB smaller whilst having +1% extra accuracy vs the QAT version!**

2. Efficiency wise, 2bit Q2_K_XL and others seem to do very well!


Quant

Unsloth

Unsloth + QAT

Disk Size

Efficiency

IQ1_M

48.10

47.23

6.51

3.42

IQ2_XXS

59.20

56.57

7.31

4.32

IQ2_M

66.47

64.47

8.96

4.40

Q2_K_XL

68.70

67.77

9.95

4.30

Q3_K_XL

70.87

69.50

12.76

3.49

**Q4_K_XL**

**71.47**

**71.07**

**15.64**

**2.94**

**Google QAT**

**70.64**

**17.2**

**2.65**

Click here for Full Google's Gemma 3 (27B) QAT Benchmarks: [Direct link to heading](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs#click-here-for-full-googles-gemma-3-27b-qat-benchmarks)

Model

Unsloth

Unsloth + QAT

Disk Size

Efficiency

IQ1_S

41.87

43.37

6.06

3.03

IQ1_M

48.10

47.23

6.51

3.42

IQ2_XXS

59.20

56.57

7.31

4.32

IQ2_M

66.47

64.47

8.96

4.40

Q2_K

68.50

67.60

9.78

4.35

Q2_K_XL

68.70

67.77

9.95

4.30

IQ3_XXS

68.27

67.07

10.07

4.18

Q3_K_M

70.70

69.77

12.51

3.58

Q3_K_XL

70.87

69.50

12.76

3.49

Q4_K_M

71.23

71.00

15.41

2.98

**Q4_K_XL**

**71.47**

**71.07**

**15.64**

**2.94**

Q5_K_M

71.77

71.23

17.95

2.58

Q6_K

71.87

71.60

20.64

2.26

Q8_0

71.60

71.53

26.74

1.74

**Google QAT**

**70.64**

**17.2**

**2.65**

## [Direct link to heading](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs#llama-4-bug-fixes--run)    🦙 Llama 4 Bug Fixes + Run

We also helped and fixed a few Llama 4 bugs:

- Llama 4 Scout changed the RoPE Scaling configuration in their official repo. We helped resolve issues in llama.cpp to enable this [change here](https://github.com/ggml-org/llama.cpp/pull/12889)









![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-7ff8229dfa96425f50c2c87f9ca988ef9cc99eff%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=5e8f69d3&sv=2)

- Llama 4's QK Norm's epsilon for both Scout and Maverick should be from the config file - this means using 1e-05 and not 1e-06. We helped resolve these in [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/12889) and [transformers](https://github.com/huggingface/transformers/pull/37418)

- The Llama 4 team and vLLM also independently fixed an issue with QK Norm being shared across all heads (should not be so) [here](https://github.com/vllm-project/vllm/pull/16311). MMLU Pro increased from 68.58% to 71.53% accuracy.

- [Wolfram Ravenwolf](https://x.com/WolframRvnwlf/status/1909735579564331016) showcased how our GGUFs via llama.cpp attain much higher accuracy than third party inference providers - this was most likely a combination of the issues explained above, and also probably due to quantization issues.









![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-76c49d8c8e3e42f7407f431a2cede369f87878e4%252FGoC79hYXwAAPTMs.jpg%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=8f9607fb&sv=2)


As shown in our graph, our 4-bit Dynamic QAT quantization deliver better performance on 5-shot MMLU while also being smaller in size.

### [Direct link to heading](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs#running-llama-4-scout)    Running Llama 4 Scout:

To run Llama 4 Scout for example, first clone llama.cpp:

Copy

```
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

Then download out new dynamic v 2.0 quant for Scout:

Copy

```
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
    local_dir = "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
    allow_patterns = ["*IQ2_XXS*"],
)
```

And and let's do inference!

Copy

```
./llama.cpp/llama-cli \
    --model unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/Llama-4-Scout-17B-16E-Instruct-UD-IQ2_XXS.gguf \
    --threads 32 \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --seed 3407 \
    --prio 3 \
    --temp 0.6 \
    --min-p 0.01 \
    --top-p 0.9 \
    -no-cnv \
    --prompt "<|header_start|>user<|header_end|>\n\nCreate a Flappy Bird game.<|eot|><|header_start|>assistant<|header_end|>\n\n"
```

Read more on running Llama 4 here: [https://docs.unsloth.ai/basics/tutorial-how-to-run-and-fine-tune-llama-4](https://docs.unsloth.ai/basics/tutorial-how-to-run-and-fine-tune-llama-4)

[PreviousText-to-Speech Fine-tuning](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) [NextAider Polyglot Benchmarks](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)

Last updated 12 days ago

Was this helpful?

This site uses cookies to deliver its service and to analyze traffic. By browsing this site, you accept the [privacy policy](https://policies.gitbook.com/privacy/cookies).

AcceptReject