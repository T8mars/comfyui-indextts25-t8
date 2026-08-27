"""Compatibility shim for the former automatic download-source detector.

Custom-node import and inference must not perform unsolicited network probes or
read process-wide proxy switches.  The explicit model downloader selects
ModelScope or Hugging Face from its ``--source`` argument; normal inference
uses Hugging Face only when an auxiliary file is genuinely missing.
"""


def need_proxy(timeout: float = 3.0) -> bool:
    """Return the safe default without opening sockets during import/inference."""

    del timeout
    return False
