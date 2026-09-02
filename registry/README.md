# Public source-backed model metadata

This directory is the transparent metadata mirror for exact Public VRhino
model identities that use local upstream acquisition and conversion. Entries
follow Runnable Model Package schema 1 and contain no model weights, converted
components, cache objects, credentials, or test media.

The release binary carries the corresponding fixed converter specifications.
The public files here let users inspect the immutable source identities,
hashes, product declaration, workflow configuration and license boundary.

The frozen v0.6.0-alpha Public successor identities are:

- `vrhino/ltx-video-v0.9.1:1.1.1`
- `vrhino/wan2.1-t2v-1.3b:1.0.1`
- `vrhino/mochi-1-preview:1.0.1`
- `vrhino/musetalk-v1.5:1.0.1`
- `vrhino/latentsync-1.6:1.0.1`

Historical package directories remain immutable and available. Successors are
additive exact versions; this repository has no mutable `latest` or default
pointer to retarget.

Registry presence does not by itself claim that a compatible binary release
has already been published.
