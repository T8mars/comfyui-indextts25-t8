# Repository instructions

## GitHub publishing

- Before every `git push` to GitHub, ensure `[project].version` in `pyproject.toml`
  is higher than the version on the target remote branch.
- Unless the outgoing commits already contain an intentional version change, run
  `python scripts/bump_version.py patch`, stage `pyproject.toml`, and include the
  version update in the outgoing commit.
- Use `patch` for fixes and repository-only maintenance, `minor` for backward-compatible
  features, and `major` for breaking changes.
- Never reuse a version that has been uploaded to Comfy Registry; published Registry
  versions are immutable.
- Before pushing a release commit, run `python -m pytest` and `comfy node validate`
  when Comfy CLI is available.
