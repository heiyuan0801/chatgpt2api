# 0001: Image generation AI upscaling

Status: pending-verification

- Owner: Image Task and Studio maintainers
- Created: 2026-10-03
- Updated: 2026-10-03
- Related issues/ADRs: `docs/adr/0002-studio-uses-asynchronous-image-tasks.md`

## Problem and user outcome

Generated images may be returned below the requested delivery size. A caller or
Studio user can request the original result, 2K, or 4K AI upscaling while the
existing Image Task, storage, logging, OpenAI-compatible routes, and Studio
flow remain authoritative.

## Current behavior and evidence

Image generation and edit requests are executed by `ImageTaskService` and the
OpenAI protocol services, then stored by `ImageStorageService`. Studio polls
the asynchronous task projection described by ADR 0002. The new request fields
are optional and preserve the existing original-image default.

## Scope

- In scope: `upscale` and `upscale_target` request fields, Final2x-core
  execution, aspect-ratio-preserving 2K/4K sizing, original retention, task
  progress, API responses, and Studio selection.
- In scope: Docker/runtime dependency and timeout configuration.

## Non-goals

- Not included: a second image-generation pipeline, Electron GUI integration,
  or replacement of Image Task or Image Asset ownership.

## Domain language

“AI upscale” is the delivery stage after the upstream image has been fetched
and before the final Image Asset URL is returned. “4K” means a longest edge of
3840 pixels while preserving the source aspect ratio.

## Unique owners

| Concern | Authoritative owner | Interface consumed by others | Forbidden mirror or fallback |
| --- | --- | --- | --- |
| Upscale sizing and engine execution | `services/image_upscale_service.py` | `upscale_image` result | Router-specific subprocess calls |
| Task lifecycle and progress | `ImageTaskService` | task projection | Frontend task state as truth |
| Image persistence | `ImageStorageService` | stored asset URL/path | Overwriting the original asset |

## Backend Modules and Interfaces

The protocol services parse the optional fields into `ConversationRequest`.
`format_image_result` invokes the upscale service, stores the original before
the transformed asset, and returns the transformed URL and dimensions. Async
task records persist the selected mode and expose the two upscale stages.

## Frontend interaction and responsive behavior

Studio keeps its existing upstream ratio/resolution controls and adds an
“输出增强” choice for 原图, 2K 高清, and 4K 高清. The selected value is sent
with both generation and edit task requests; task polling displays the backend
stage label.

## Persistence impact

The original and transformed files are both retained by the existing image
storage boundary. Image Task JSON gains the request selection and no new
database tables are introduced.

## Concurrency, security, and failure behavior

Final2x runs under the existing image attempt boundary with a process lock and
configurable timeout. A local upscale failure is a delivery failure and does
not switch accounts or retry upstream generation. Existing authorization and
owner-scoped task polling remain unchanged.

## Acceptance criteria

- [ ] Original, 2K, and 4K selections produce the expected final dimensions
      without stretching.
- [x] The original asset remains addressable when the transformed asset is
      returned.
- [x] Final2x failures are reported as delivery failures without account
      switching.
- [x] Existing requests without `upscale` retain their current behavior.
- [x] Backend compilation and frontend build pass; Docker verification is
      recorded separately when a daemon is available.

## Verification matrix

| Acceptance criterion | Verification level | Evidence required | Status |
| --- | --- | --- | --- |
| Aspect-ratio sizing | unit/manual | Pillow dimension probe | complete |
| Existing default path | manual | `format_image_result` probe | complete |
| API/task contract | compile/build | Python compile and Vue build | complete |
| Final2x in Docker | integration | CPU engine benchmark and live `/v1/images/generations` request | complete |

## Documentation and CHANGELOG impact

Update the critical-flow map, deployment runbook, upstream reference catalogue,
and `CHANGELOG.md` with the new delivery stage and runtime settings.

## ADR requirement

Not required: the implementation stays within the Image Task and Image Asset
owners already established by ADR 0002 and the existing storage boundary.

## Rollback

Set `upscale=false` for callers and select the existing `sharp_lanczos3` or
`pillow_lanczos` engine in settings; reverting the code removes only the new
optional stage and its dependency.

## Unresolved questions

- Different source images and concurrent CPU requests still need load testing;
  the verified 1536×1024 live request completed in about 82 seconds.
