# Open Source Contributions to OpenVINO contrib

Contributions to the openvino_contrib repository (openvino-notes module).

## ON-001-01: Account-bound token access and reset
[PR #1168](https://github.com/openvinotoolkit/openvino_contrib/pull/1168) · ✅ Merged

- Added mandatory `AccountKey` to `AccessTokenProvider` methods.
- Mismatched account returns denial without session access; resetting token A preserves B.
- Masked token value in `toString` and user-facing `IdentityService`.
- Added test coverage in `DriveCloudComponentTest` and `GoogleIdentityControllerTest`.

## ON-002-01: Discard stale session login results
[PR #1200](https://github.com/openvinotoolkit/openvino_contrib/pull/1200) · ⏳ Pending

- Bound login and permission results to session generation and `AccountKey`.
- Logout and account switch increment generation and clear permission state.
- Late permissions for A do not authorize B; canceled coroutines are not treated as completed logins.
- Added concurrency tests with controlled wait points (no `sleep`).
