"""A failing method left on disk without the protocol (no --list output).
Must be scored as failed, not silently skipped."""

def advantages(rewards):
    return [0.0 for _ in rewards]
