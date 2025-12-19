"""
filepydantic - Memory-mapped Pydantic models backed by JSON files.

Your Pydantic model IS a file. Changes to the model save to disk.
Changes to the file reload the model.

Vendored from https://github.com/jacobdejean/filepydantic
"""

from me.vendor.filepydantic.core import FileModel, FileDirectory

__all__ = ["FileModel", "FileDirectory"]
__version__ = "0.1.0"
