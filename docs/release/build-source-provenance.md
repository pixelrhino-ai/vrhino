# v0.6.0-alpha Build-Source Provenance Boundary

The production release builder for v0.6.0-alpha has been qualified to build
offline from repository-local, pinned dependency sources. Its tokenizer chain
uses these exact upstream identities:

| Source | Exact identity | License |
| --- | --- | --- |
| tokenizers-cpp | `c586c52f93f7b060753bd2388eb96a105cb7374d` | Apache-2.0 |
| SentencePiece | `e0f0f966959108415183d6cbe7a9051ca8bc2da1` | Apache-2.0 |
| msgpack-c | `092bc69b6e815980bce7808595c914dd3a29f905` | BSL-1.0 |
| Abseil | `255c84dadd029fd8ad25c5efb5933e47beaa00c7` | Apache-2.0 |

The Rust tokenizer build uses the locked 80-crate registry graph and local
source replacement qualified by the release builder. Cargo is invoked with
locked and offline resolution.

At the v0.6.0-alpha tag, the Public repository was a release-facing contract projection and did not contain the full tokenizer build-source closure. Current main now contains the full production source and pinned vendored build inputs. See [source build and tests](../source-build.md) for current build instructions. This change does not alter the historical release, tag or binary assets.

The vendored build-source payload is not placed in the binary runtime package.
Runtime third-party notices and license texts remain inventoried in
[`licenses/THIRD_PARTY_NOTICES.txt`](../../licenses/THIRD_PARTY_NOTICES.txt).
