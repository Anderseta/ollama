# Security

This policy covers the WildLlama fork at
[Anderseta/wildllama](https://github.com/Anderseta/wildllama). WildLlama is
maintained independently of Ollama; the upstream Ollama team is not responsible
for reviewing this fork's changes.

## Reporting a vulnerability

Do not post credentials, exploit instructions or private model data in a public
issue. Check the repository's
[Security tab](https://github.com/Anderseta/wildllama/security) for private
vulnerability reporting. If **Report a vulnerability** is available, use it to
send a confidential report to this repository's maintainers.

Private reporting depends on the repository's GitHub settings. If that option is
unavailable, use a private contact method published by the repository owner to
arrange confidential disclosure before sending sensitive details. This project
does not publish a dedicated security email address or a guaranteed response
time.

Include the affected commit/build, platform, minimal reproduction, expected and
actual behavior, likely impact and any mitigation you have identified. Keep
large or sensitive artifacts out of the initial report.

For a vulnerability reproduced in unmodified upstream Ollama, also follow
[Ollama's security policy](https://github.com/ollama/ollama/blob/main/SECURITY.md).
Clearly distinguish an upstream issue from a WildLlama-specific change.

## Running a local inference service

Keep inference endpoints restricted to trusted clients and review authentication
and network exposure before hosting a service. Treat models, templates and
extensions as inputs from their publishers, and keep the runtime and GPU drivers
updated. RAM and VRAM quotas describe the documented allocation scopes; they are
not security sandbox boundaries.
