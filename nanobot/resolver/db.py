"""EntityDB — SQLite-backed entity store with incremental CRUD and platform sync."""

import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import httpx

DB_PATH = Path.home() / ".nanobot" / "cache" / "symbols.db"
CACHE_TTL = 86400  # 24 hours


class EntityDB:
    """SQLite entity store supporting incremental adds and platform sync."""

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._last_sync_check: float = 0

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.path))

    # ── Read ──────────────────────────────────────────

    def get_symbol(self, symbol: str) -> dict | None:
        if not self.path.exists():
            return None
        db = self._connect()
        try:
            row = db.execute(
                "SELECT symbol, name, type, exchange, country, currency FROM symbols WHERE symbol=?",
                (symbol,),
            ).fetchone()
            if not row:
                return None
            return {
                "symbol": row[0], "name": row[1], "type": row[2],
                "exchange": row[3], "country": row[4], "currency": row[5],
            }
        finally:
            db.close()

    def count(self) -> dict:
        if not self.path.exists():
            return {"symbols": 0, "aliases": 0}
        db = self._connect()
        try:
            syms = db.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
            aliases = db.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
            return {"symbols": syms, "aliases": aliases}
        finally:
            db.close()

    def get_version(self) -> str | None:
        if not self.path.exists():
            return None
        db = self._connect()
        try:
            return self._get_meta(db, "version")
        finally:
            db.close()

    def set_version(self, version: str):
        db = self._connect()
        try:
            self._set_meta(db, "version", version)
        finally:
            db.close()

    # ── Incremental Write ─────────────────────────────

    def add_symbols(self, symbols: list[dict]):
        """Upsert symbols. Each dict: {symbol, name, type, exchange?, country?, currency?, delisted?}."""
        if not symbols:
            return
        db = self._connect()
        try:
            self._ensure_tables(db)
            db.executemany(
                "INSERT OR REPLACE INTO symbols (symbol, name, type, exchange, country, currency, delisted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        s["symbol"], s["name"], s["type"],
                        s.get("exchange"), s.get("country"), s.get("currency"),
                        int(s.get("delisted", False)),
                    )
                    for s in symbols
                ],
            )
            db.commit()
        finally:
            db.close()

    def add_aliases(self, aliases: list[tuple[str, str]]):
        """Add (alias, symbol) pairs. Upserts."""
        if not aliases:
            return
        db = self._connect()
        try:
            self._ensure_tables(db)
            db.executemany(
                "INSERT OR REPLACE INTO aliases (alias, symbol) VALUES (?, ?)",
                aliases,
            )
            db.commit()
        finally:
            db.close()

    def remove_symbols(self, symbols: list[str]):
        """Remove symbols and their aliases (CASCADE)."""
        if not symbols:
            return
        db = self._connect()
        try:
            placeholders = ",".join("?" * len(symbols))
            db.execute(f"DELETE FROM aliases WHERE symbol IN ({placeholders})", symbols)
            db.execute(f"DELETE FROM symbols WHERE symbol IN ({placeholders})", symbols)
            db.commit()
        finally:
            db.close()

    # ── Bulk Load (for trie building) ─────────────────

    def iter_symbols(self) -> list[dict]:
        """Load all symbols for trie construction."""
        if not self.path.exists():
            return []
        db = self._connect()
        try:
            try:
                cols = [r[1] for r in db.execute("PRAGMA table_info(symbols)").fetchall()]
                has_delisted = "delisted" in cols
            except Exception:
                has_delisted = False

            query = (
                "SELECT symbol, name, type, exchange, country, currency"
                + (", delisted" if has_delisted else "")
                + " FROM symbols"
            )
            entries = []
            for row in db.execute(query):
                entries.append({
                    "symbol": row[0], "name": row[1], "type": row[2],
                    "exchange": row[3], "country": row[4], "currency": row[5],
                    "delisted": bool(row[6]) if has_delisted and len(row) > 6 else False,
                })
            return entries
        finally:
            db.close()

    def iter_aliases(self) -> dict[str, list[str]]:
        """Load alias → [symbols] mapping."""
        if not self.path.exists():
            return {}
        db = self._connect()
        try:
            alias_to_symbols: dict[str, list[str]] = {}
            for row in db.execute("SELECT alias, symbol FROM aliases"):
                alias_to_symbols.setdefault(row[0].lower(), []).append(row[1])
            return alias_to_symbols
        finally:
            db.close()

    # ── Platform Sync ─────────────────────────────────

    def sync_from_platform(self, api_key: str, api_base: str) -> bool:
        """Check platform for new version, download if needed. Returns True if updated."""
        if time.time() - self._last_sync_check < CACHE_TTL:
            return False
        self._last_sync_check = time.time()

        try:
            local_version = self.get_version()
            headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"}
            if local_version:
                headers["If-None-Match"] = f'"{local_version}"'

            resp = httpx.get(f"{api_base}/symbols/dictionary", headers=headers, timeout=15)
            if resp.status_code == 304:
                return False
            if resp.status_code != 200:
                return False

            data = resp.json()
            remote_version = data.get("version", "")
            download_url = data.get("url", "")
            if not download_url:
                return False
            if local_version and local_version == remote_version:
                return False

            self._download_snapshot(download_url, remote_version)
            return True
        except Exception:
            return False

    def _download_snapshot(self, url: str, version: str):
        """Download SQLite from R2, atomically replace local copy."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".db", dir=str(self.path.parent))
        try:
            os.close(fd)
            with httpx.stream("GET", url, timeout=60) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_bytes(8192):
                        f.write(chunk)

            db = sqlite3.connect(tmp_path)
            count = db.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
            if count == 0:
                db.close()
                os.unlink(tmp_path)
                return
            self._set_meta(db, "last_sync", str(time.time()))
            if not self._get_meta(db, "version"):
                self._set_meta(db, "version", version)
            db.close()
            shutil.move(tmp_path, str(self.path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── Internal helpers ──────────────────────────────

    @staticmethod
    def _get_meta(db: sqlite3.Connection, key: str) -> str | None:
        try:
            row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row[0] if row else None
        except sqlite3.OperationalError:
            return None

    @staticmethod
    def _set_meta(db: sqlite3.Connection, key: str, value: str):
        db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        db.commit()

    @staticmethod
    def _ensure_tables(db: sqlite3.Connection):
        db.execute("""CREATE TABLE IF NOT EXISTS symbols (
            symbol   TEXT PRIMARY KEY,
            name     TEXT NOT NULL,
            type     TEXT NOT NULL,
            exchange TEXT,
            country  TEXT,
            currency TEXT,
            delisted INTEGER DEFAULT 0
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS aliases (
            alias   TEXT NOT NULL,
            symbol  TEXT NOT NULL,
            PRIMARY KEY (symbol, alias),
            FOREIGN KEY (symbol) REFERENCES symbols(symbol) ON DELETE CASCADE
        )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_alias ON aliases(alias)")
        db.commit()
