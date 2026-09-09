# Public main governance v1

Public `pixelrhino-ai/vrhino` main is the only production source of truth. Normal changes follow Public main → topic branch → PR → required Public CI → squash merge → Public main. Follow [CONTRIBUTING.md](../CONTRIBUTING.md).

## Main protection

The active repository ruleset [Public main protection v1](https://github.com/pixelrhino-ai/vrhino/rules/22619082) targets only `refs/heads/main`. It does not target release tags. There is no redundant classic branch protection rule.

| Setting | Policy |
| --- | --- |
| Force pushes / main deletion | Blocked |
| Pull request before merge | Required |
| Approvals | 0; no external reviewer required |
| Code owner / last-push approval / stale approval dismissal | Disabled |
| Required check | `Linux host and offline tokenizer build` |
| Check producer | GitHub Actions, integration ID `15368` |
| Current-base validation | Required; update the topic branch if main advances |
| Linear history | Required for future changes; existing history stays intact |
| Signed commits | Not required |
| Bypass actors | None, including no configured admin bypass |
| Merge methods | Squash enabled; merge commits and rebase merging disabled |
| Auto-merge / automatic branch deletion | Disabled |

Squash merging gives each production PR one focused commit. The initial audit found no open PRs and no merge commits in the available history, so this choice preserves a simple history without disrupting pending work. Contributors may delete merged topic branches manually; main deletion remains blocked.

Administrators can edit repository rules. Any administrative relaxation or bypass is **EMERGENCY ONLY**, never the normal development workflow. Record the reason and exact affected SHAs, restore protection promptly and verify CI. Emergency authority never permits rewriting production history or mutating published releases.

## CI contract

The existing [Public source workflow](../.github/workflows/source.yml) runs on pull requests (including those targeting main), pushes to main and manual dispatch. Its stable job name above is the required check, discovered from a successful check run on Public main; the workflow display name is not the check context.

CI runs on Ubuntu 22.04 with CUDA disabled and tokenizers enabled. It validates source hygiene and public contracts, then configures and builds with the network isolated and runs the public host CTest profile. Inputs are synthetic or public repository inputs; no private weights, media, evidence or GPU are required. Installing developer prerequisites requires network access before the offline configure/build boundary. Host CI does not qualify GPU execution or real-model behavior.

Keep this check meaningful and running for every PR to main, including documentation changes. Do not replace it with a skipped or placeholder check. A future rename must coordinate the real workflow check and ruleset so that protection never points at a missing check or admits unvalidated changes.

## Maintainer and release boundaries

`CODEOWNERS_NOT_NEEDED_V1`: the audit found one effective maintainer. CODEOWNERS would add no practical ownership clarity, so none is added and code owner approval is not required.

Published releases and tags are immutable by project policy. `v0.5.0-alpha`, `v0.6.0-alpha` and `v0.7.0-alpha` remain frozen. Do not move published tags, replace binary assets, force-update release refs or reuse versions; fixes require a new version. This branch ruleset does not enforce tag or asset immutability, and the existing releases are not marked immutable by GitHub. Maintainers must preserve the project policy.

Private research starts from a recorded Public SHA, stays in the separate research workspace and enters production only as a clean diff through Public PRs and CI. Local evidence, artifacts, builds and archives stay outside Public Git. Research history is not production history.
