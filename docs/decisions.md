# Design decisions

Corrections to the original plan, recorded as they are found.

## 1. Digest day uses the arXiv announcement date, not `published`

The arXiv API's `published` field is the submission time of v1, not the day the paper was
announced on the listing. Submissions close daily at 14:00 US Eastern, so a paper submitted on
Friday after the cutoff only appears in Tuesday's listing (later around holidays). Eligibility for a digest is therefore decided by announcement date:
the digest for day D contains the papers first announced on D. Revisions (v2+) update the
stored record but never enter a digest.

## 2. Hugging Face Space runs Streamlit through the Docker SDK (superseded by 4)

Streamlit is no longer a built-in Space SDK
([docs](https://huggingface.co/docs/hub/spaces-sdks-streamlit)). The Space is created with
the Docker SDK and the Streamlit template. Nothing else in the plan changes.

## 3. Embedding model and fusion

`BAAI/bge-small-en-v1.5` (384 dimensions) on CPU, with `chromadb`
pinned in `requirements.txt` so stored embeddings stay comparable across runs. Vectors are
L2-normalised and queries get bge's retrieval instruction prefix. Dense and BM25 rankings (top 50
each) are fused with reciprocal rank fusion, k = 60. Changing the model means re-embedding the
whole store.

## 4. App hosted on Streamlit Community Cloud, embeddings through ONNX Runtime

Since July 2026 Hugging Face requires a PRO plan to host Docker and Gradio Spaces on the free
`cpu-basic` hardware (the first deploy failed with HTTP 402; static Spaces are still free). The app
moved to Streamlit Community Cloud, which is free and deploys from GitHub. The Dataset on Hugging
Face is unchanged and stays the source of truth.

To fit the free tier's memory, `sentence-transformers` and torch were replaced by the model's
official ONNX export (`onnx/model.onnx` in `BAAI/bge-small-en-v1.5`) run with ONNX Runtime: same
weights, same CLS pooling and normalisation, so vectors already stored stay comparable and nothing
is re-embedded. `onnxruntime` and `tokenizers` were already installed as `chromadb` dependencies.

## 5. "Most voted" uses Hugging Face Daily Papers upvotes

Requested after the MVP: a list of the most popular papers for a period chosen by the user, the same
for everyone. arXiv exposes no views or ratings, and no other open source publishes view counts, so
the signal is the upvote count on Hugging Face Daily Papers. It is added as its own tab next to the
profile digest instead of replacing it, so the profile digest and its evaluation (Milestone 6) stay
as they are. Popularity needs no relevance judge: the vote count is the measurement. Caveats: Daily
Papers covers only papers someone submitted (a few dozen a day, all areas of AI), and votes favour
well-known labs.

## 6. Digest profile changed to "advances in current frontier models"

On 2026-10-27 the profile in `profile.md` moved from RAG and information retrieval to advances in
current frontier models (releases, architectures, scaling, post-training, reasoning, multimodal,
efficiency, capability benchmarks). Each digest now records a `profile_id` (a hash of the profile
text) and `profile_name`. Judge verdicts are cached per profile, and precision@5 only counts digests
built with the current profile, so the earlier RAG digests are left out of the metric instead of being
judged against interests they were not picked for.
