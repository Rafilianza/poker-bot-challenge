"""Preserve registered source files and their attribution when exporting."""


def lean_public_files(files):
    """Retain the original API without stripping licenses, data, or source text.

    Registration already excludes caches and enforces archive size limits.
    A source-reference marker is not permission to remove attribution.
    """
    return dict(files)
