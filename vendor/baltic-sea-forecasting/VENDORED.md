# Vendored source — do not edit in place

| | |
|---|---|
| **Origin** | https://github.com/joeloskarsson/baltic-sea-forecasting |
| **Commit** | `9c8c874adb228ef5dfbd45a34e9f00b3130d8417` |
| **Branch** | `main` |
| **Commit date** | 2026-08-13 |
| **Commit message** | Global model (#5) |
| **Vendored on** | 2026-08-14 |
| **Removed** | `.git/`, `.github/`, `figures/` |

## Why this is a copy and not a submodule

The repo is **private on GitHub**, and the GitLab CI runner has no GitHub credentials
(`.gitlab-ci.yml` uses `GIT_SUBMODULE_STRATEGY: recursive`). A submodule would fail the
pipeline checkout. The repo is also frozen, so there is no upstream to track.

## Scope — build-time only

This tree is **not** part of the deployed backend. It is not in the Docker build context, not
copied into the image, and its dependencies are not in `backend/requirements.txt`. It exists to
(1) script the oceanography models to TorchScript and (2) serve as the reference for
replicating preprocessing in `backend/src`.

⚠️ It ships its own `neural_lam/` package that **collides** with the `backend/neural-lam`
submodule. Importing this at backend runtime silently loads the wrong model code.

**Full rationale and rules: [`VENDORED_BALTIC_SEA_FORECASTING.md`](../../VENDORED_BALTIC_SEA_FORECASTING.md) (repo root).**

## Re-vendoring

Do not patch this tree. If upstream ever changes, re-vendor from scratch and update the commit
SHA above, so this stays a faithful snapshot of a known commit.

```bash
git clone https://github.com/joeloskarsson/baltic-sea-forecasting.git /tmp/bsf
git -C /tmp/bsf log -1 --format='%H%n%cI%n%s'
rm -rf /tmp/bsf/.git /tmp/bsf/.github /tmp/bsf/figures
rm -rf vendor/baltic-sea-forecasting
mv /tmp/bsf vendor/baltic-sea-forecasting
# then restore this file with the new SHA
```
