# Design decisions

Corrections to the original plan, recorded as they are found.

## 1. Digest day uses the arXiv announcement date, not `published`

The arXiv API's `published` field is the submission time of v1, not the day the paper was
announced on the listing. Submissions close daily at 14:00 US Eastern, so a paper submitted on
Friday after the cutoff only appears in Tuesday's listing (later around holidays). Eligibility for a digest is therefore decided by announcement date:
the digest for day D contains the papers first announced on D. Revisions (v2+) update the
stored record but never enter a digest.

## 2. Hugging Face Space runs Streamlit through the Docker SDK

Streamlit is no longer a built-in Space SDK
([docs](https://huggingface.co/docs/hub/spaces-sdks-streamlit)). The Space is created with
the Docker SDK and the Streamlit template. Nothing else in the plan changes.
