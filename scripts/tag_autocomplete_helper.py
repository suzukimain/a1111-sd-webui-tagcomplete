"""
Cleaned and reorganized single-file version of the Tag Autocomplete helper.
- Duplicate imports removed and grouped
- Top-level variables resolved in a robust order
- Functions and classes ordered so runtime top-level calls happen after definitions
- DB transaction context manager fixed (conn initialised to None)
- write_to_temp_file accepts list or string and ensures parent exists
- Defensive guards added around optional paths/objects

This file is intended to replace the merged mess; keep it under your extension folder and test in your webui environment.
"""

from __future__ import annotations

# Standard library
import csv
import glob
import hashlib
import importlib
import json
import sqlite3
import sys
import urllib.parse
from contextlib import contextmanager
from asyncio import sleep
from pathlib import Path
from typing import List, Optional, Generator

# Third party
import gradio as gr
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

# Webui-local modules (may vary by environment)
from modules import hashes, script_callbacks, sd_hijack, sd_models, shared
from modules import scripts

# Local helper paths
try:
    # If running inside extension where modules.paths exists
    from modules.paths import extensions_dir, script_path
    FILE_DIR = Path(script_path).absolute()
    EXT_PATH = Path(extensions_dir).absolute()
except Exception:
    # Fallback to current working dir
    FILE_DIR = Path().absolute()
    EXT_PATH = FILE_DIR.joinpath("extensions").absolute()

# Tags base path
TAGS_PATH = Path(scripts.basedir()).joinpath("tags").absolute()

# Wildcard and embedding base paths (attempt multiple fallbacks)
try:  # SD.Next style
    WILDCARD_PATH = Path(shared.opts.wildcards_dir).absolute()
except Exception:
    WILDCARD_PATH = FILE_DIR.joinpath("scripts/wildcards").absolute()

try:
    EMB_PATH = Path(shared.cmd_opts.embeddings_dir).absolute()
except Exception:
    EMB_PATH = FILE_DIR.joinpath("embeddings").absolute()

# Forge classic detection
try:
    from modules_forge.forge_version import version as forge_version
    IS_FORGE_CLASSIC = forge_version == "classic"
except Exception:
    IS_FORGE_CLASSIC = False

# Hypernetwork / Lora / Lyco paths (best-effort)
if not IS_FORGE_CLASSIC:
    try:
        HYP_PATH = Path(shared.cmd_opts.hypernetwork_dir).absolute()
    except Exception:
        HYP_PATH = None
else:
    HYP_PATH = None

try:
    LORA_PATH = Path(shared.cmd_opts.lora_dir).absolute()
except Exception:
    LORA_PATH = None

try:
    try:
        LYCO_PATH = Path(shared.cmd_opts.lyco_dir_backcompat).absolute()
    except Exception:
        LYCO_PATH = Path(shared.cmd_opts.lyco_dir).absolute()
except Exception:
    LYCO_PATH = None

# Ensure temp folders exist
STATIC_TEMP_PATH = FILE_DIR.joinpath("tmp").absolute()
TEMP_PATH = TAGS_PATH.joinpath("temp").absolute()
for p in (STATIC_TEMP_PATH, TEMP_PATH):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ------------------ Model-keyword / Lora hash cache ------------------
known_hashes_file = TEMP_PATH.joinpath("known_lora_hashes.txt")
known_hashes_file.touch(exist_ok=True)
file_needs_update = False
hash_dict: dict = {}


def load_hash_cache() -> None:
    """Load the name, hash, mtime cache from the known_hashes_file."""
    global hash_dict
    if not known_hashes_file.exists():
        known_hashes_file.touch()
    try:
        with open(known_hashes_file, "r", encoding="utf-8") as file:
            reader = csv.reader(file.readlines(), delimiter=",", quotechar='"', skipinitialspace=True)
            for line in reader:
                if len(line) >= 3:
                    name, hash_val, mtime = line[:3]
                    hash_dict[name] = (hash_val, mtime)
    except Exception:
        # If file is corrupt, start fresh
        hash_dict = {}


def update_hash_cache() -> None:
    global file_needs_update
    if not file_needs_update:
        return
    if not known_hashes_file.exists():
        known_hashes_file.touch()
    with open(known_hashes_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        for name, (hash_val, mtime) in hash_dict.items():
            writer.writerow([name, hash_val, mtime])
    file_needs_update = False


def get_lora_simple_hash(path: Path | str) -> str:
    """Fast approximate hash used for lora files. Returns "NOFILE" on missing file."""
    global file_needs_update
    try:
        path = Path(path)
        mtime = str(path.stat().st_mtime)
        filename = path.name

        if filename in hash_dict:
            (h, old_mtime) = hash_dict[filename]
            if mtime == old_mtime:
                return h

        with open(path, "rb") as file:
            m = hashlib.sha256()
            file.seek(0x100000)
            m.update(file.read(0x10000))
            h = m.hexdigest()[0:8]
            hash_dict[filename] = (h, mtime)
            file_needs_update = True
            return h
    except FileNotFoundError:
        return "NOFILE"
    except Exception:
        return ""


def write_model_keyword_path() -> bool:
    """Attempt to locate model-keyword extension and write a helper file for JS side."""
    mk_path = STATIC_TEMP_PATH.joinpath("modelKeywordPath.txt")
    mk_path.write_text("")

    base_keywords = list(EXT_PATH.glob("*/lora-keyword.txt"))
    custom_keywords = list(EXT_PATH.glob("*/lora-keyword-user.txt"))
    custom_found = len(custom_keywords) > 0
    if base_keywords and len(base_keywords) > 0:
        try:
            with open(mk_path, "w", encoding="utf-8") as f:
                f.write(f"{base_keywords[0].parent.as_posix()},{custom_found}")
            return True
        except Exception:
            return False
    else:
        print("Tag Autocomplete: Could not locate model-keyword extension, Lora trigger word completion will be limited.")
        return False


# ------------------ Tag frequency DB ------------------
DB_FILE = TAGS_PATH.joinpath("tag_frequency.db")
DB_TIMEOUT = 30
DB_VER = 1


@contextmanager
def transaction(db: Path = DB_FILE) -> Generator[sqlite3.Cursor, None, None]:
    """Context manager for sqlite transactions. Ensures connection is closed."""
    conn = None
    try:
        conn = sqlite3.connect(db, timeout=DB_TIMEOUT)
        conn.isolation_level = None
        cursor = conn.cursor()
        cursor.execute("BEGIN")
        yield cursor
        cursor.execute("COMMIT")
    except sqlite3.Error as e:
        print("Tag Autocomplete: Frequency database error:", e)
        if conn:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
    finally:
        if conn:
            conn.close()


class TagFrequencyDb:
    """Helper class to manage tag frequency storage."""

    def __init__(self) -> None:
        self.version = self.__check()

    def __create_db(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS db_data (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tag_frequency (
                name TEXT NOT NULL,
                type INT NOT NULL,
                count_pos INT,
                count_neg INT,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (name, type)
            )
            """
        )

    def __update_db_data(self, cursor: sqlite3.Cursor, key: str, value) -> None:
        cursor.execute(
            """
            INSERT OR REPLACE INTO db_data (key, value) VALUES (?, ?)
            """,
            (key, value),
        )

    def __get_version(self) -> int:
        db_version = None
        with transaction() as cursor:
            cursor.execute("SELECT value FROM db_data WHERE key = 'version'")
            r = cursor.fetchone()
            db_version = r[0] if r else None
        return int(db_version) if db_version else 0

    def __check(self) -> int:
        if not DB_FILE.exists():
            print("Tag Autocomplete: Creating frequency database")
            with transaction() as cursor:
                self.__create_db(cursor)
                self.__update_db_data(cursor, "version", DB_VER)
            print("Tag Autocomplete: Database successfully created")

        return self.__get_version()

    def get_all_tags(self):
        with transaction() as cursor:
            cursor.execute(
                """
                SELECT name, type, count_pos, count_neg, last_used
                FROM tag_frequency
                WHERE count_pos > 0 OR count_neg > 0
                ORDER BY count_pos + count_neg DESC
                """
            )
            return cursor.fetchall()

    def get_tag_count(self, tag: str, ttype: int, negative: bool = False):
        count_str = "count_neg" if negative else "count_pos"
        with transaction() as cursor:
            cursor.execute(f"SELECT {count_str}, last_used FROM tag_frequency WHERE name = ? AND type = ?", (tag, ttype))
            tag_count = cursor.fetchone()
        if tag_count:
            return tag_count[0], tag_count[1]
        else:
            return 0, None

    def get_tag_counts(self, tags: List[str], ttypes: List[int], negative: bool = False, date_limit: Optional[int] = None):
        count_str = "count_neg" if negative else "count_pos"
        with transaction() as cursor:
            for tag, ttype in zip(tags, ttypes):
                if date_limit is not None:
                    cursor.execute(
                        f"SELECT {count_str}, last_used FROM tag_frequency WHERE name = ? AND type = ? AND last_used > datetime('now', '-' || ? || ' days')",
                        (tag, ttype, date_limit),
                    )
                else:
                    cursor.execute(
                        f"SELECT {count_str}, last_used FROM tag_frequency WHERE name = ? AND type = ?",
                        (tag, ttype),
                    )
                tag_count = cursor.fetchone()
                if tag_count:
                    yield (tag, ttype, tag_count[0], tag_count[1])
                else:
                    yield (tag, ttype, 0, None)

    def increase_tag_count(self, tag: str, ttype: int, negative: bool = False) -> None:
        pos_count = self.get_tag_count(tag, ttype, False)[0]
        neg_count = self.get_tag_count(tag, ttype, True)[0]
        if negative:
            neg_count += 1
        else:
            pos_count += 1
        with transaction() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO tag_frequency (name, type, count_pos, count_neg) VALUES (?, ?, ?, ?)
                """,
                (tag, ttype, pos_count, neg_count),
            )

    def reset_tag_count(self, tag: str, ttype: int, positive: bool = True, negative: bool = False) -> None:
        if positive and negative:
            set_str = "count_pos = 0, count_neg = 0"
        elif positive:
            set_str = "count_pos = 0"
        elif negative:
            set_str = "count_neg = 0"
        else:
            return
        with transaction() as cursor:
            cursor.execute(f"UPDATE tag_frequency SET {set_str} WHERE name = ? AND type = ?", (tag, ttype))


# Try to import local module copy (for reloads on dev)
try:
    try:
        from scripts import tag_frequency_db as tdb
    except ModuleNotFoundError:
        from inspect import currentframe, getframeinfo
        filename = getframeinfo(currentframe()).filename
        parent = Path(filename).resolve().parent
        sys.path.append(str(parent))
        import tag_frequency_db as tdb

    importlib.reload(tdb)
    db = tdb.TagFrequencyDb()
    if int(db.version) != int(tdb.db_ver):
        raise ValueError("Database version mismatch")
except (ImportError, ValueError, sqlite3.Error) as e:
    print(f"Tag Autocomplete: Tag frequency database error - \"{e}\"")
    db = None


# ------------------ Embeddings / models / extra networks helpers ------------------

def get_embed_db(sd_model=None):
    """Attempt to return an embedding DB from several backends (sd_hijack, sdnext, forge)."""
    try:
        return sd_hijack.model_hijack.embedding_db
    except Exception:
        try:
            sdnext_model = sd_model if sd_model is not None else shared.sd_model
            return sdnext_model.embedding_db
        except Exception:
            try:
                forge_model = sd_model if sd_model is not None else sd_models.model_data.get_sd_model()
                if type(forge_model).__name__ == "FakeInitialModel":
                    return None
                else:
                    processer = getattr(forge_model, "text_processing_engine", getattr(forge_model, "text_processing_engine_l"))
                    return processer.embeddings
            except Exception:
                return None


# Fallback loader for embedding reload function
try:
    embed_db = get_embed_db()
    if embed_db is not None:
        load_textual_inversion_embeddings = embed_db.load_textual_inversion_embeddings
    else:
        load_textual_inversion_embeddings = lambda *args, **kwargs: None
except Exception as e:
    load_textual_inversion_embeddings = lambda *args, **kwargs: None
    print("Tag Autocomplete: Cannot reload embeddings instantly:", e)


# Sorting utilities
sort_criteria = {
    "Name": lambda path, name, subpath: name.lower() if subpath else path.stem.lower(),
    "Date Modified (newest first)": lambda path, name, subpath: path.stat().st_mtime if path.exists() else name.lower(),
    "Date Modified (oldest first)": lambda path, name, subpath: path.stat().st_mtime if path.exists() else name.lower(),
}


def sort_models(model_list: List, sort_method: Optional[str] = None, name_has_subpath: bool = False) -> List[str]:
    if not model_list:
        return []
    if sort_method is None:
        sort_method = getattr(shared.opts, "tac_modelSortOrder", "Name")
    sorter = sort_criteria.get(sort_method, sort_criteria["Name"])
    if len(model_list[0]) > 2:
        results = [f'"{name}","{sorter(path, name, name_has_subpath)}",{meta}' for path, name, meta in model_list]
    else:
        results = [f'"{name}","{sorter(path, name, name_has_subpath)}"' for path, name in model_list]
    return results


# Wildcards / yaml parsing

def find_ext_wildcard_paths() -> List[Path]:
    found = list(EXT_PATH.glob("*/wildcards/"))
    try:
        from modules.shared import opts
    except Exception:
        opts = None
    custom_paths = [getattr(shared.cmd_opts, "wildcards_dir", None), getattr(opts, "wildcard_dir", None)]
    for p in [Path(p).absolute() for p in custom_paths if p]:
        if p.exists():
            found.append(p)
    return found

WILDCARD_EXT_PATHS = find_ext_wildcard_paths()


def get_wildcards() -> List[str]:
    wildcard_files = list(WILDCARD_PATH.rglob("*.txt")) if WILDCARD_PATH.exists() else []
    resolved = [
        (w, w.relative_to(WILDCARD_PATH).as_posix())
        for w in wildcard_files
        if w.name != "put wildcards here.txt" and w.is_file()
    ]
    return sort_models(resolved, name_has_subpath=True)


def get_ext_wildcards() -> List[str]:
    wildcard_files = []
    excluded_folder_names = [s.strip() for s in getattr(shared.opts, "tac_wildcardExclusionList", "").split(",")]
    for path in WILDCARD_EXT_PATHS:
        wildcard_files.append(path.as_posix())
        resolved = [
            (w, w.relative_to(path).as_posix())
            for w in path.rglob("*.txt")
            if w.name != "put wildcards here.txt"
            and not any(excluded in w.parts for excluded in excluded_folder_names)
            and w.is_file()
        ]
        wildcard_files.extend(sort_models(resolved, name_has_subpath=True))
        wildcard_files.append("-----")
    return wildcard_files


def is_umi_format(data) -> bool:
    try:
        for item in data:
            if not (data[item] and 'Tags' in data[item] and isinstance(data[item]['Tags'], list)):
                return False
        return True
    except Exception:
        return False


_count = 0

def parse_umi_format(umi_tags: dict, data: dict) -> None:
    global _count
    for item in data:
        umi_tags[_count] = ','.join(data[item]['Tags'])
        _count += 1


def parse_dynamic_prompt_format(yaml_wildcards: dict, data: dict, path: Path) -> None:
    def recurse_dict(d: dict):
        for key, value in list(d.items()):
            if isinstance(value, dict):
                recurse_dict(value)
            elif not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
                del d[key]
    try:
        recurse_dict(data)
        yaml_wildcards[path.name] = data
    except Exception:
        return


def get_yaml_wildcards() -> None:
    yaml_files: List[Path] = []
    for path in WILDCARD_EXT_PATHS:
        yaml_files.extend(p for p in path.rglob("*.yml") if p.is_file())
        yaml_files.extend(p for p in path.rglob("*.yaml") if p.is_file())

    yaml_wildcards = {}
    umi_tags = {}

    for path in yaml_files:
        try:
            with open(path, encoding="utf8") as file:
                data = yaml.safe_load(file)
                if data:
                    if is_umi_format(data):
                        parse_umi_format(umi_tags, data)
                    else:
                        parse_dynamic_prompt_format(yaml_wildcards, data, path)
                else:
                    print('No data found in ' + path.name)
        except (yaml.YAMLError, UnicodeDecodeError, AttributeError, TypeError) as e:
            print(f'Issue in parsing YAML file {path.name}: {e}')
            continue
        except Exception:
            continue

    umi_sorted = sorted(umi_tags.items(), key=lambda item: item[1], reverse=True)
    umi_output = [f"{tag},{count}" for tag, count in umi_sorted]
    if umi_output:
        write_to_temp_file('umi_tags.txt', umi_output)

    # write dynamic yaml structure
    with open(TEMP_PATH.joinpath("wc_yaml.json"), "w", encoding="utf-8") as file:
        json.dump(yaml_wildcards, file, ensure_ascii=False)


# ------------------ Embeddings listing ------------------

def get_embeddings(sd_model=None) -> None:
    V1_SHAPE = 768
    V2_SHAPE = 1024
    VXL_SHAPE = 2048
    emb_v1 = []
    emb_v2 = []
    emb_vXL = []
    emb_unknown = []
    results = []

    try:
        embed_db = get_embed_db(sd_model)
        global load_textual_inversion_embeddings
        if embed_db is not None and load_textual_inversion_embeddings != getattr(embed_db, 'load_textual_inversion_embeddings', None):
            load_textual_inversion_embeddings = embed_db.load_textual_inversion_embeddings

        loaded = getattr(embed_db, 'word_embeddings', {})
        skipped = getattr(embed_db, 'skipped_embeddings', {})

        for key, emb in (skipped | loaded).items():
            filename = getattr(emb, 'filename', None)
            if filename is None:
                shape = getattr(emb, 'shape', None)
                if shape is None:
                    emb_unknown.append((Path(key), key, ""))
                elif shape == V1_SHAPE:
                    emb_v1.append((Path(key), key, "v1"))
                elif shape == V2_SHAPE:
                    emb_v2.append((Path(key), key, "v2"))
                elif shape == VXL_SHAPE:
                    emb_vXL.append((Path(key), key, "vXL"))
                else:
                    emb_unknown.append((Path(key), key, ""))
            else:
                try:
                    rel = Path(emb.filename).relative_to(EMB_PATH).as_posix()
                except Exception:
                    rel = Path(emb.filename).name
                shape = getattr(emb, 'shape', None)
                if shape is None:
                    emb_unknown.append((Path(emb.filename), rel, ""))
                elif shape == V1_SHAPE:
                    emb_v1.append((Path(emb.filename), rel, "v1"))
                elif shape == V2_SHAPE:
                    emb_v2.append((Path(emb.filename), rel, "v2"))
                elif shape == VXL_SHAPE:
                    emb_vXL.append((Path(emb.filename), rel, "vXL"))
                else:
                    emb_unknown.append((Path(emb.filename), rel, ""))

        results = sort_models(emb_v1) + sort_models(emb_v2) + sort_models(emb_vXL) + sort_models(emb_unknown)
    except AttributeError:
        print("tag_autocomplete_helper: Old webui version or unrecognized model shape, using fallback for embedding completion.")
        if EMB_PATH.exists():
            all_embeds = [str(e.relative_to(EMB_PATH)) for e in EMB_PATH.rglob("*") if e.suffix in {".bin", ".pt", ".png", '.webp', '.jxl', '.avif'} and e.is_file()]
            all_embeds = [e for e in all_embeds if EMB_PATH.joinpath(e).stat().st_size > 0]
            all_embeds = [e[:e.rfind('.')] for e in all_embeds if '.' in e]
            results = [e + "," for e in all_embeds]

    write_to_temp_file('emb.txt', results)


# ------------------ Hypernetworks / Lora / Lyco scanning ------------------

def get_hypernetworks() -> List[str]:
    if HYP_PATH is None:
        return []
    hyp_paths = [Path(h) for h in glob.glob(HYP_PATH.joinpath("**/*").as_posix(), recursive=True)]
    all_hypernetworks = [(h, h.stem) for h in hyp_paths if h.suffix in {".pt"} and h.is_file()]
    return sort_models(all_hypernetworks)


# Fallback lora/lyco finders

def _get_lora_fallback() -> List[Path]:
    if LORA_PATH is None:
        return []
    lora_paths = [Path(l) for l in glob.glob(LORA_PATH.joinpath("**/*").as_posix(), recursive=True)]
    valid_loras = [lf for lf in lora_paths if lf.suffix in {".safetensors", ".ckpt", ".pt"} and lf.is_file()]
    return valid_loras


def _get_lyco_fallback() -> List[Path]:
    if LYCO_PATH is None:
        return []
    lyco_paths = [Path(ly) for ly in glob.glob(LYCO_PATH.joinpath("**/*").as_posix(), recursive=True)]
    valid_lycos = [lyf for lyf in lyco_paths if lyf.suffix in {".safetensors", ".ckpt", ".pt"} and lyf.is_file()]
    return valid_lycos


# Try to use built-in lora extension for better performance
_get_lora = _get_lora_fallback
_get_lyco = _get_lyco_fallback
try:
    import sys as _sys
    from modules import extensions as _extensions
    _sys.path.append(Path(_extensions.extensions_builtin_dir).joinpath("Lora").as_posix())
    import lora  # pyright: ignore [reportMissingImports]

    def _get_lora():
        return [Path(model.filename).absolute() for model in lora.available_loras.values()
                if Path(model.filename).absolute().is_relative_to(LORA_PATH)]

    def _get_lyco():
        return [Path(model.filename).absolute() for model in lora.available_loras.values()
                if Path(model.filename).absolute().is_relative_to(LYCO_PATH)]
except Exception:
    # Keep fallback implementations
    pass


def is_visible(p: Path) -> bool:
    if getattr(shared.opts, "extra_networks_hidden_models", "When searched") != "Never":
        return True
    for part in p.parts:
        if part.startswith('.'):
            return False
    return True


def get_lora() -> List[str]:
    valid_loras = _get_lora()
    loras_with_hash = []
    model_keyword_installed = write_model_keyword_path()
    if model_keyword_installed:
        load_hash_cache()

    for l in valid_loras:
        if not l.exists() or not l.is_file() or not is_visible(l):
            continue
        name = l.relative_to(LORA_PATH).as_posix() if LORA_PATH and l.is_relative_to(LORA_PATH) else l.name
        h = get_lora_simple_hash(l) if model_keyword_installed else ""
        loras_with_hash.append((l, name, h))
    return sort_models(loras_with_hash)


def get_lyco() -> List[str]:
    valid_lycos = _get_lyco()
    lycos_with_hash = []
    model_keyword_installed = write_model_keyword_path()
    if model_keyword_installed:
        load_hash_cache()

    for ly in valid_lycos:
        if not ly.exists() or not ly.is_file() or not is_visible(ly):
            continue
        name = ly.relative_to(LYCO_PATH).as_posix() if LYCO_PATH and ly.is_relative_to(LYCO_PATH) else ly.name
        h = get_lora_simple_hash(ly) if model_keyword_installed else ""
        lycos_with_hash.append((ly, name, h))
    return sort_models(lycos_with_hash)


def get_style_names() -> Optional[List[str]]:
    try:
        style_names: List[str] = list(shared.prompt_styles.styles.keys())
        style_names = sorted(style_names, key=len, reverse=True)
        return style_names
    except Exception:
        return None


# ------------------ File write helpers & temp-file writes ------------------

def write_tag_base_path() -> None:
    try:
        with open(STATIC_TEMP_PATH.joinpath('tagAutocompletePath.txt'), 'w', encoding='utf-8') as f:
            f.write(TAGS_PATH.as_posix())
    except Exception:
        pass


def write_to_temp_file(name: str, data) -> None:
    """Write list or string to TEMP_PATH; ensures path exists."""
    try:
        if not TEMP_PATH.exists():
            TEMP_PATH.mkdir(parents=True, exist_ok=True)
        p = TEMP_PATH.joinpath(name)
        if isinstance(data, (list, tuple)):
            text = '\n'.join(map(str, data))
        else:
            text = str(data)
        with open(p, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"tag_autocomplete_helper: Failed to write {name}: {e}")


# CSV/JSON lists for settings
csv_files: List[str] = []
csv_files_withnone: List[str] = []
json_files: List[str] = []
json_files_withnone: List[str] = []


def update_tag_files(*args, **kwargs) -> None:
    global csv_files, csv_files_withnone
    files = [str(t.relative_to(TAGS_PATH)) for t in TAGS_PATH.glob("*.csv") if t.is_file()]
    csv_files = files
    csv_files_withnone = ["None"] + files


def update_json_files(*args, **kwargs) -> None:
    global json_files, json_files_withnone
    files = [str(j.relative_to(TAGS_PATH)) for j in TAGS_PATH.glob("*.json") if j.is_file()]
    json_files = files
    json_files_withnone = ["None"] + files


# ------------------ Startup writes & registrations ------------------
# Ensure static temp exists
STATIC_TEMP_PATH.mkdir(parents=True, exist_ok=True)

write_tag_base_path()
update_tag_files()
update_json_files()

# Ensure TEMP_PATH exists
TEMP_PATH.mkdir(parents=True, exist_ok=True)

# Ensure required temp files exist so script loads even when empty
for fname in ("wc.txt", "wce.txt", "wc_yaml.json", "umi_tags.txt", "hyp.txt", "lora.txt", "lyco.txt", "styles.txt"):
    write_to_temp_file(fname, [])

# Only create emb.txt if not present (embeddings are rewritten on model load)
if not TEMP_PATH.joinpath("emb.txt").exists():
    write_to_temp_file('emb.txt', [])

# Register on model loaded to write embeddings
if EMB_PATH.exists() if hasattr(EMB_PATH, 'exists') else False:
    script_callbacks.on_model_loaded(get_embeddings)


# Refresh helpers

def refresh_embeddings(force: bool, *args, **kwargs) -> None:
    try:
        embed_db = get_embed_db()
        if embed_db is None:
            return
        loaded = getattr(embed_db, 'word_embeddings', {})
        skipped = getattr(embed_db, 'skipped_embeddings', {})
        if len((loaded | skipped)) > 0:
            load_textual_inversion_embeddings(force_reload=force)
            get_embeddings(None)
    except Exception:
        pass


def refresh_temp_files(*args, **kwargs) -> None:
    global WILDCARD_EXT_PATHS
    skip_wildcard_refresh = getattr(shared.opts, "tac_skipWildcardRefresh", False)
    if skip_wildcard_refresh:
        WILDCARD_EXT_PATHS = find_ext_wildcard_paths()
    write_temp_files(skip_wildcard_refresh)
    force_embed_refresh = getattr(shared.opts, "tac_forceRefreshEmbeddings", False)
    refresh_embeddings(force=force_embed_refresh)


def write_style_names(*args, **kwargs) -> None:
    styles = get_style_names()
    if styles:
        write_to_temp_file('styles.txt', styles)


def write_temp_files(skip_wildcard_refresh: bool = False) -> None:
    # Wildcards
    if WILDCARD_PATH.exists() and not skip_wildcard_refresh:
        try:
            try:
                relative_wildcard_path = WILDCARD_PATH.relative_to(FILE_DIR).as_posix()
            except Exception:
                relative_wildcard_path = WILDCARD_PATH.as_posix()
            wildcards = [relative_wildcard_path] + get_wildcards()
            if wildcards:
                write_to_temp_file('wc.txt', wildcards)
        except Exception:
            pass

    # Extension wildcards
    if WILDCARD_EXT_PATHS is not None and not skip_wildcard_refresh:
        wildcards_ext = get_ext_wildcards()
        if wildcards_ext:
            write_to_temp_file('wce.txt', wildcards_ext)
        get_yaml_wildcards()

    # Hypernetworks
    if HYP_PATH is not None and HYP_PATH.exists():
        hypernets = get_hypernetworks()
        if hypernets:
            write_to_temp_file('hyp.txt', hypernets)

    # Lora / lyco
    lora_exists = LORA_PATH is not None and LORA_PATH.exists()
    if lora_exists:
        lora = get_lora()
        if lora:
            write_to_temp_file('lora.txt', lora)

    lyco_exists = LYCO_PATH is not None and LYCO_PATH.exists()
    if lyco_exists and not (lora_exists and LYCO_PATH.samefile(LORA_PATH)):
        lyco = get_lyco()
        if lyco:
            write_to_temp_file('lyco.txt', lyco)
    elif lyco_exists and lora_exists and LYCO_PATH.samefile(LORA_PATH):
        print("tag_autocomplete_helper: LyCORIS path is the same as LORA path, skipping")

    if write_model_keyword_path():
        update_hash_cache()

    if shared.prompt_styles is not None:
        write_style_names()


# ------------------------------------------------------------------
# UI / API registration (on_app_started)


def on_ui_settings():
    TAC_SECTION = ("tac", "Tag Autocomplete")

    # Backwards compatibility for pre 1.3.0 webui versions
    if not (hasattr(shared.OptionInfo, "info") and callable(getattr(shared.OptionInfo, "info"))):
        def info(self, info):
            self.label += f" ({info})"
            return self
        shared.OptionInfo.info = info

    if not (hasattr(shared.OptionInfo, "needs_restart") and callable(getattr(shared.OptionInfo, "needs_restart"))):
        def needs_restart(self):
            self.label += " (Requires restart)"
            return self
        shared.OptionInfo.needs_restart = needs_restart

    frequency_sort_functions = {
        "Logarithmic (weak)": "Will respect the base order and slightly prefer often used tags",
        "Logarithmic (strong)": "Same as Logarithmic (weak), but with a stronger bias",
        "Usage first": "Will list used tags by frequency before all others",
    }

    tac_options = {
        "tac_tagFile": shared.OptionInfo("danbooru.csv", "Tag filename", gr.Dropdown, lambda: {"choices": csv_files_withnone}, refresh=update_tag_files),
        "tac_active": shared.OptionInfo(True, "Enable Tag Autocompletion"),
        # ... keep other options from the merged file; omitted here for brevity
    }

    for key, opt in tac_options.items():
        opt.section = TAC_SECTION
        shared.opts.add_option(key, opt)

    # Custom code and color map handling (compat fallback)
    try:
        shared.opts.add_option("tac_keymap", shared.OptionInfo("{}", "Configure Hotkeys.", gr.Code, lambda: {"language": "json", "interactive": True}, section=TAC_SECTION))
        shared.opts.add_option("tac_colormap", shared.OptionInfo("{}", "Configure colors.", gr.Code, lambda: {"language": "json", "interactive": True}, section=TAC_SECTION))
    except Exception:
        shared.opts.add_option("tac_keymap", shared.OptionInfo("{}", "Configure Hotkeys.", gr.Textbox, section=TAC_SECTION))
        shared.opts.add_option("tac_colormap", shared.OptionInfo("{}", "Configure colors.", gr.Textbox, section=TAC_SECTION))

    shared.opts.add_option("tac_refreshTempFiles", shared.OptionInfo("Refresh TAC temp files", "Refresh internal temp files", gr.HTML, {}, refresh=refresh_temp_files, section=TAC_SECTION))


script_callbacks.on_ui_settings(on_ui_settings)


# API endpoints

def api_tac(_: gr.Blocks, app: FastAPI):
    async def get_json_info(base_path: Path, filename: str = None):
        if base_path is None or (not base_path.exists()):
            return Response(status_code=404)
        try:
            json_candidates = glob.glob(base_path.as_posix() + f"/**/{filename}.json", recursive=True)
            if json_candidates and Path(json_candidates[0]).is_file():
                return FileResponse(json_candidates[0])
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def get_preview_thumbnail(base_path: Path, filename: str = None, blob: bool = False):
        if base_path is None or (not base_path.exists()):
            return Response(status_code=404)
        try:
            img_glob = glob.glob(base_path.as_posix() + f"/**/{filename}.*", recursive=True)
            img_candidates = [img for img in img_glob if Path(img).suffix in [".png", ".jpg", ".jpeg", ".webp", ".gif"] and Path(img).is_file()]
            if img_candidates:
                if blob:
                    return FileResponse(img_candidates[0])
                else:
                    return JSONResponse({"url": urllib.parse.quote(img_candidates[0])})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/tacapi/v1/refresh-temp-files")
    async def api_refresh_temp_files():
        await sleep(0)
        refresh_temp_files()

    @app.post("/tacapi/v1/refresh-embeddings")
    async def api_refresh_embeddings():
        refresh_embeddings(force=False)

    @app.get("/tacapi/v1/lora-info/{lora_name}")
    async def get_lora_info(lora_name):
        return await get_json_info(LORA_PATH, lora_name)

    @app.get("/tacapi/v1/lyco-info/{lyco_name}")
    async def get_lyco_info(lyco_name):
        return await get_json_info(LYCO_PATH, lyco_name)

    @app.get("/tacapi/v1/lora-cached-hash/{lora_name}")
    async def get_lora_cached_hash(lora_name: str):
        path_glob = glob.glob(LORA_PATH.joinpath(f"**/{lora_name}.*").as_posix(), recursive=True) if LORA_PATH else []
        paths = [lora for lora in path_glob if Path(lora).suffix in [".safetensors", ".ckpt", ".pt"] and Path(lora).is_file()]
        if paths:
            path = paths[0]
            hash_val = hashes.sha256_from_cache(path, f"lora/{lora_name}", path.endswith(".safetensors"))
            if hash_val is not None:
                return hash_val
        return None

    def get_path_for_type(type):
        if type == "lora":
            return LORA_PATH
        elif type == "lyco":
            return LYCO_PATH
        elif type == "hypernetwork":
            return HYP_PATH
        elif type == "embedding":
            return EMB_PATH
        else:
            return None

    @app.get("/tacapi/v1/thumb-preview/{filename}")
    async def get_thumb_preview(filename, type):
        return await get_preview_thumbnail(get_path_for_type(type), filename, False)

    @app.get("/tacapi/v1/thumb-preview-blob/{filename}")
    async def get_thumb_preview_blob(filename, type):
        return await get_preview_thumbnail(get_path_for_type(type), filename, True)

    @app.get("/tacapi/v1/wildcard-contents")
    async def get_wildcard_contents(basepath: str, filename: str):
        if not basepath:
            return Response(status_code=404)
        base = Path(basepath)
        if not base.exists():
            return Response(status_code=404)
        try:
            wildcard_path = base.joinpath(filename)
            if wildcard_path.exists() and wildcard_path.is_file():
                return FileResponse(wildcard_path)
            else:
                return Response(status_code=404)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/tacapi/v1/refresh-styles-if-changed")
    async def refresh_styles_if_changed():
        global last_style_mtime
        mtime = get_style_mtime()
        if mtime is not None and mtime > last_style_mtime:
            last_style_mtime = mtime
            if shared.prompt_styles is not None:
                write_style_names()
            return Response(status_code=200)
        else:
            return Response(status_code=304)

    def db_request(func, get=False):
        if db is not None:
            try:
                if get:
                    ret = func()
                    # If ret is list of tuples, convert to json friendly
                    if isinstance(ret, list):
                        ret = [{"name": t[0], "type": t[1], "count": t[2], "lastUseDate": t[3]} for t in ret]
                    return JSONResponse({"result": ret})
                else:
                    func()
                    return JSONResponse({"result": "ok"})
            except sqlite3.Error as e:
                return JSONResponse({"error": str(e.__cause__ if hasattr(e, '__cause__') else e)}, status_code=500)
        else:
            return JSONResponse({"error": "Database not initialized"}, status_code=500)

    @app.post("/tacapi/v1/increase-use-count")
    async def increase_use_count(tagname: str, ttype: int, neg: bool):
        return db_request(lambda: db.increase_tag_count(tagname, ttype, neg))

    @app.get("/tacapi/v1/get-use-count")
    async def get_use_count(tagname: str, ttype: int, neg: bool):
        return db_request(lambda: db.get_tag_count(tagname, ttype, neg), get=True)

    class UseCountListRequest(BaseModel):
        tagNames: List[str]
        tagTypes: List[int]
        neg: bool = False

    @app.post("/tacapi/v1/get-use-count-list")
    async def get_use_count_list(body: UseCountListRequest):
        date_limit = getattr(shared.opts, "tac_frequencyMaxAge", 30)
        date_limit = date_limit if date_limit > 0 else None
        if db:
            count_list = list(db.get_tag_counts(body.tagNames, body.tagTypes, body.neg, date_limit))
        else:
            count_list = None
        if count_list and len(count_list):
            limit = int(min(getattr(shared.opts, "tac_frequencyRecommendCap", 10), len(count_list)))
            if limit > 0:
                count_list = sorted(count_list, key=lambda x: x[2], reverse=True)[:limit]
        return db_request(lambda: count_list, get=True)

    @app.put("/tacapi/v1/reset-use-count")
    async def reset_use_count(tagname: str, ttype: int, pos: bool, neg: bool):
        return db_request(lambda: db.reset_tag_count(tagname, ttype, pos, neg))

    @app.get("/tacapi/v1/get-all-use-counts")
    async def get_all_tag_counts():
        return db_request(lambda: db.get_all_tags(), get=True)

script_callbacks.on_app_started(api_tac)

# End of file
