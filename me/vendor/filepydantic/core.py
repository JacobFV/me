"""
Core implementation of file-backed Pydantic models.
"""

from __future__ import annotations

import json
import threading
import hashlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar, Iterator

from pydantic import BaseModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

T = TypeVar("T", bound=BaseModel)


class FileModel(Generic[T]):
    """
    A Pydantic model backed by a JSON file.

    Changes to the model automatically save to disk.
    Changes to the file automatically reload the model.

    Usage:
        config = FileModel("config.json", Config, default=Config())
        config.model.count += 1  # Auto-saves
    """

    def __init__(
        self,
        path: str | Path,
        model_type: type[T],
        default: T | None = None,
        auto_save: bool = True,
        auto_watch: bool = False,
        create_dirs: bool = True,
        frozen: bool = False,
    ):
        """
        Initialize a file-backed model.

        Args:
            path: Path to the JSON file
            model_type: The Pydantic model class
            default: Default value if file doesn't exist
            auto_save: Automatically save on changes
            auto_watch: Start file watching immediately
            create_dirs: Create parent directories if needed
            frozen: If True, model cannot be modified after initial load
        """
        self._path = Path(path)
        self._model_type = model_type
        self._default = default
        self._auto_save = auto_save
        self._frozen = frozen
        self._model: T | None = None
        self._lock = threading.RLock()
        self._callbacks: list[Callable[[T, T], None]] = []
        self._observer: Observer | None = None
        self._last_hash: str | None = None
        self._last_mtime: float = 0

        if create_dirs:
            self._path.parent.mkdir(parents=True, exist_ok=True)

        # Initial load or create
        self._load_or_create()

        if auto_watch:
            self.start_watching()

    def _load_or_create(self) -> None:
        """Load from file or create with default."""
        if self._path.exists():
            self._load()
        elif self._default is not None:
            self._model = self._default.model_copy()
            self.save()
        else:
            raise FileNotFoundError(
                f"File {self._path} does not exist and no default provided"
            )

    def _load(self) -> None:
        """Load model from file."""
        with self._lock:
            content = self._path.read_text()
            data = json.loads(content)
            self._model = self._model_type.model_validate(data)
            self._last_hash = hashlib.md5(content.encode()).hexdigest()
            self._last_mtime = self._path.stat().st_mtime

    def _content_changed(self) -> bool:
        """Check if file content has changed since last load."""
        if not self._path.exists():
            return False

        try:
            mtime = self._path.stat().st_mtime
            if mtime <= self._last_mtime:
                return False

            content = self._path.read_text()
            current_hash = hashlib.md5(content.encode()).hexdigest()
            return current_hash != self._last_hash
        except Exception:
            return False

    @property
    def path(self) -> Path:
        """Get the file path."""
        return self._path

    @property
    def model(self) -> T:
        """Get the current model value."""
        with self._lock:
            if self._model is None:
                self._load()
            return self._model  # type: ignore

    @model.setter
    def model(self, value: T) -> None:
        """Set the model value."""
        if self._frozen:
            raise ValueError(f"FileModel at {self._path} is frozen and cannot be modified")

        with self._lock:
            old = self._model
            self._model = value

            if self._auto_save:
                self.save()

            self._notify_change(old, value)

    def update(self, **kwargs: Any) -> None:
        """Update specific fields of the model."""
        if self._frozen:
            raise ValueError(f"FileModel at {self._path} is frozen and cannot be modified")

        with self._lock:
            old = self._model
            # Create new model with updates
            current_data = self._model.model_dump() if self._model else {}
            current_data.update(kwargs)
            self._model = self._model_type.model_validate(current_data)

            if self._auto_save:
                self.save()

            self._notify_change(old, self._model)

    def reload(self) -> None:
        """Reload model from file."""
        with self._lock:
            old = self._model
            self._load()
            if old != self._model:
                self._notify_change(old, self._model)

    def save(self) -> None:
        """Save model to file."""
        with self._lock:
            if self._model is None:
                return

            content = json.dumps(
                self._model.model_dump(mode='json'),
                indent=2,
                default=str,
            )
            self._path.write_text(content)
            self._last_hash = hashlib.md5(content.encode()).hexdigest()
            self._last_mtime = self._path.stat().st_mtime

    def delete(self) -> None:
        """Delete the backing file."""
        with self._lock:
            if self._path.exists():
                self._path.unlink()
            self._model = None

    def exists(self) -> bool:
        """Check if the backing file exists."""
        return self._path.exists()

    def on_change(self, callback: Callable[[T, T], None]) -> None:
        """Register a callback for when the model changes."""
        self._callbacks.append(callback)

    def _notify_change(self, old: T | None, new: T | None) -> None:
        """Notify callbacks of a change."""
        if old is None or new is None:
            return
        for callback in self._callbacks:
            try:
                callback(old, new)
            except Exception:
                pass  # Don't let callback errors break the model

    # File watching

    def start_watching(self) -> None:
        """Start watching the file for external changes."""
        if self._observer is not None:
            return

        handler = _FileChangeHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._path.parent), recursive=False)
        self._observer.start()

    def stop_watching(self) -> None:
        """Stop watching the file."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    @contextmanager
    def watching(self) -> Iterator[None]:
        """Context manager for file watching."""
        self.start_watching()
        try:
            yield
        finally:
            self.stop_watching()

    def __del__(self):
        """Cleanup on deletion."""
        self.stop_watching()

    def __repr__(self) -> str:
        return f"FileModel({self._path}, {self._model_type.__name__})"


class _FileChangeHandler(FileSystemEventHandler):
    """Handler for file system events."""

    def __init__(self, file_model: FileModel):
        self._file_model = file_model
        self._debounce_time = 0.1
        self._last_event = 0.0

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory:
            return

        if Path(event.src_path) != self._file_model._path:
            return

        # Debounce rapid events
        import time
        now = time.time()
        if now - self._last_event < self._debounce_time:
            return
        self._last_event = now

        # Only reload if content actually changed
        if self._file_model._content_changed():
            self._file_model.reload()


class FileDirectory:
    """
    A directory of file-backed Pydantic models.

    Manages multiple related FileModel instances in a directory.
    Access models by attribute name.

    Usage:
        body = FileDirectory("./agent/")
        body.register("config", Config, default=Config())
        body.register("state", State)

        body.config.model.debug = True
        print(body.state.model.status)
    """

    def __init__(
        self,
        path: str | Path,
        create: bool = True,
        auto_watch: bool = False,
    ):
        """
        Initialize a file directory.

        Args:
            path: Path to the directory
            create: Create directory if it doesn't exist
            auto_watch: Start file watching for all registered models
        """
        self._path = Path(path)
        self._auto_watch = auto_watch
        self._models: dict[str, FileModel] = {}

        if create:
            self._path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Get the directory path."""
        return self._path

    def register(
        self,
        name: str,
        model_type: type[T],
        default: T | None = None,
        frozen: bool = False,
        filename: str | None = None,
    ) -> FileModel[T]:
        """
        Register a file-backed model.

        Args:
            name: Attribute name for accessing the model
            model_type: The Pydantic model class
            default: Default value if file doesn't exist
            frozen: If True, model cannot be modified after creation
            filename: Custom filename (default: {name}.json)

        Returns:
            The created FileModel
        """
        filename = filename or f"{name}.json"
        file_path = self._path / filename

        file_model = FileModel(
            path=file_path,
            model_type=model_type,
            default=default,
            auto_watch=self._auto_watch,
            frozen=frozen,
        )

        self._models[name] = file_model
        return file_model

    def get(self, name: str) -> FileModel | None:
        """Get a registered model by name."""
        return self._models.get(name)

    def __getattr__(self, name: str) -> Any:
        """Access model by attribute."""
        if name.startswith('_'):
            raise AttributeError(name)

        model = self._models.get(name)
        if model is not None:
            return model.model
        raise AttributeError(f"No model registered with name '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set model value by attribute."""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return

        model = self._models.get(name)
        if model is not None:
            model.model = value
            return

        object.__setattr__(self, name, value)

    def file(self, name: str) -> FileModel | None:
        """Get the FileModel wrapper (not just the model value)."""
        return self._models.get(name)

    def list(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def start_watching(self) -> None:
        """Start watching all files."""
        for model in self._models.values():
            model.start_watching()

    def stop_watching(self) -> None:
        """Stop watching all files."""
        for model in self._models.values():
            model.stop_watching()

    @contextmanager
    def watching(self) -> Iterator[None]:
        """Context manager for watching all files."""
        self.start_watching()
        try:
            yield
        finally:
            self.stop_watching()

    def ensure_subdirs(self, *subdirs: str) -> None:
        """Create subdirectories within this directory."""
        for subdir in subdirs:
            (self._path / subdir).mkdir(parents=True, exist_ok=True)

    def subdir(self, name: str) -> "FileDirectory":
        """Get or create a subdirectory as a new FileDirectory."""
        path = self._path / name
        return FileDirectory(path, create=True, auto_watch=self._auto_watch)

    def __repr__(self) -> str:
        return f"FileDirectory({self._path}, models={list(self._models.keys())})"
