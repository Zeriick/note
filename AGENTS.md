# Notes Repository Instructions

These instructions apply to the entire notes repository.

## Writing principles

- Write notes as reusable, topic-centered knowledge that can be understood independently of the conversation that created them.
- Default to general concepts, definitions, mechanisms, examples, trade-offs, and limitations that transfer across papers and projects.
- Do not mention an active review, manuscript number, paper ID, one-off benchmark result, transient discussion, or current task context unless the user explicitly asks for source-specific notes.
- When knowledge comes from a particular paper, generalize the stable technical content into the main note. Put paper-specific claims, parameters, experimental results, and critiques in a clearly labeled source-specific section or a separate note only when requested.
- Distinguish universal principles from implementation-specific conventions. State explicitly when graph direction, terminology, IR mapping, or solver behavior varies across tools.
- Define acronyms on first use, include a small concrete example where useful, and record important correctness assumptions and failure modes.
- Prefer Chinese prose for new notes unless the user requests another language. Keep standard technical terms in English when that improves precision.
- Preserve the existing Obsidian Markdown style and repository organization. Use `[[wikilinks]]` when linking to another note in the repository.

## Updating existing notes

- Preserve useful general material and existing user-authored content.
- Remove conversation-specific framing when converting a temporary explanation into a durable note.
- Do not silently turn a general note into a review of one paper or tool.
- If a source-specific detail is necessary to explain a general mechanism, label it as an example rather than presenting it as the universal definition.

