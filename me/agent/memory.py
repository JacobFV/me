"""
Semantic memory system for agent runs.

Tracks all agent interactions and provides semantic search
for retrieving relevant memories and their neighbors.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb


@dataclass
class MemoryEntry:
    """A single memory entry."""

    id: str
    content: str
    tags: list[str]
    agent_id: str
    run_id: str
    timestamp: str
    metadata: dict[str, Any]


class Memory:
    """
    Semantic memory system using ChromaDB for vector storage.

    Features:
    - Semantic search across all agent memories
    - Tag-based filtering
    - Neighbor retrieval (messages near a given memory)
    - Run-based organization
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistent storage
        self._client = chromadb.PersistentClient(
            path=str(storage_dir / "chroma"),
        )

        # Main collection for memories
        self._collection = self._client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},
        )

        # Index for tracking run sequences
        self._run_sequences: dict[str, list[str]] = {}
        self._load_run_sequences()

    def _load_run_sequences(self) -> None:
        """Load run sequences from disk."""
        seq_file = self.storage_dir / "run_sequences.json"
        if seq_file.exists():
            with open(seq_file, "r") as f:
                self._run_sequences = json.load(f)

    def _save_run_sequences(self) -> None:
        """Save run sequences to disk."""
        seq_file = self.storage_dir / "run_sequences.json"
        with open(seq_file, "w") as f:
            json.dump(self._run_sequences, f)

    async def store(
        self,
        content: str,
        tags: list[str],
        agent_id: str,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a memory with semantic indexing.

        Returns the memory ID.
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Build metadata
        full_metadata = {
            "agent_id": agent_id,
            "run_id": run_id,
            "timestamp": timestamp,
            "tags": ",".join(tags),  # ChromaDB requires string metadata
            **(metadata or {}),
        }

        # Add to collection
        self._collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[full_metadata],
        )

        # Track in run sequence
        if run_id not in self._run_sequences:
            self._run_sequences[run_id] = []
        self._run_sequences[run_id].append(memory_id)
        self._save_run_sequences()

        return memory_id

    async def recall(
        self,
        query: str,
        limit: int = 5,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recall memories by semantic similarity.

        Returns memories sorted by relevance with their scores.
        """
        # Build where clause for filtering
        where = {}
        if agent_id:
            where["agent_id"] = agent_id
        if run_id:
            where["run_id"] = run_id

        # Query the collection
        results = self._collection.query(
            query_texts=[query],
            n_results=limit,
            where=where if where else None,
        )

        # Format results
        memories = []
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                doc = results["documents"][0][i] if results["documents"] else ""
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0

                # Convert distance to similarity score (cosine distance -> similarity)
                score = 1 - dist

                # Filter by tags if specified
                if tags:
                    mem_tags = meta.get("tags", "").split(",")
                    if not any(t in mem_tags for t in tags):
                        continue

                memories.append({
                    "id": memory_id,
                    "content": doc,
                    "score": score,
                    "agent_id": meta.get("agent_id"),
                    "run_id": meta.get("run_id"),
                    "timestamp": meta.get("timestamp"),
                    "tags": meta.get("tags", "").split(","),
                })

        return memories

    async def get_neighbors(
        self,
        memory_id: str,
        window: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Get neighboring memories from the same run.

        Returns memories before and after the given memory in sequence.
        """
        # Find which run this memory belongs to
        target_run_id = None
        position = -1

        for run_id, sequence in self._run_sequences.items():
            if memory_id in sequence:
                target_run_id = run_id
                position = sequence.index(memory_id)
                break

        if target_run_id is None or position == -1:
            return []

        # Get neighbor IDs
        sequence = self._run_sequences[target_run_id]
        start = max(0, position - window)
        end = min(len(sequence), position + window + 1)
        neighbor_ids = sequence[start:end]

        # Fetch the memories
        results = self._collection.get(ids=neighbor_ids)

        memories = []
        if results["ids"]:
            for i, mid in enumerate(results["ids"]):
                doc = results["documents"][i] if results["documents"] else ""
                meta = results["metadatas"][i] if results["metadatas"] else {}
                memories.append({
                    "id": mid,
                    "content": doc,
                    "agent_id": meta.get("agent_id"),
                    "run_id": meta.get("run_id"),
                    "timestamp": meta.get("timestamp"),
                    "tags": meta.get("tags", "").split(","),
                    "is_target": mid == memory_id,
                    "position": sequence.index(mid),
                })

        return sorted(memories, key=lambda m: m["position"])

    async def get_run_history(
        self,
        run_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get all memories from a specific run in order."""
        sequence = self._run_sequences.get(run_id, [])
        if not sequence:
            return []

        ids_to_fetch = sequence[:limit] if limit else sequence
        results = self._collection.get(ids=ids_to_fetch)

        memories = []
        if results["ids"]:
            for i, mid in enumerate(results["ids"]):
                doc = results["documents"][i] if results["documents"] else ""
                meta = results["metadatas"][i] if results["metadatas"] else {}
                memories.append({
                    "id": mid,
                    "content": doc,
                    "agent_id": meta.get("agent_id"),
                    "run_id": meta.get("run_id"),
                    "timestamp": meta.get("timestamp"),
                    "tags": meta.get("tags", "").split(","),
                    "position": sequence.index(mid),
                })

        return sorted(memories, key=lambda m: m["position"])

    async def list_runs(
        self,
        agent_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List recent runs with summary information."""
        runs = []

        for run_id, sequence in self._run_sequences.items():
            if not sequence:
                continue

            # Get first memory to extract metadata
            first_result = self._collection.get(ids=[sequence[0]])
            if not first_result["ids"]:
                continue

            meta = first_result["metadatas"][0] if first_result["metadatas"] else {}

            # Filter by agent if specified
            if agent_id and meta.get("agent_id") != agent_id:
                continue

            runs.append({
                "run_id": run_id,
                "agent_id": meta.get("agent_id"),
                "started_at": meta.get("timestamp"),
                "memory_count": len(sequence),
            })

        # Sort by start time, most recent first
        runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return runs[:limit]

    async def delete_run(self, run_id: str) -> bool:
        """Delete all memories from a run."""
        sequence = self._run_sequences.get(run_id)
        if not sequence:
            return False

        # Delete from collection
        self._collection.delete(ids=sequence)

        # Remove from sequences
        del self._run_sequences[run_id]
        self._save_run_sequences()

        return True

    def stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        total_memories = self._collection.count()
        total_runs = len(self._run_sequences)

        return {
            "total_memories": total_memories,
            "total_runs": total_runs,
            "storage_dir": str(self.storage_dir),
        }
