import os
import json
import time
from datetime import datetime


def make_backup(path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.backup.{timestamp}.json"
    try:
        with open(path, "rb") as src, open(backup_path, "wb") as dst:
            dst.write(src.read())
        return backup_path
    except Exception as e:
        return None


def normalize_content(content):
    """
    Normalize loaded JSON content to a list that should represent the data entries.
    Rules:
      - If content is a dict and contains a 'data' key whose value is a list -> return that list
      - Elif content is a list -> return content unchanged
      - Elif content is a dict where all top-level keys are numeric strings -> return list(dict.values())
      - Else: return None to indicate not convertible by rules
    """
    if isinstance(content, dict):
        if "data" in content and isinstance(content["data"], list):
            return content["data"]
        # numeric-key dict (e.g., {"0": {...}, "1": {...}})
        if all(isinstance(k, str) and k.isdigit() for k in content.keys()):
            # sort by numeric key to keep order
            items = [content[k] for k in sorted(content.keys(), key=lambda x: int(x))]
            return items
        # sometimes nested under other key names? we won't try heuristics by default
        return None
    elif isinstance(content, list):
        return content
    else:
        return None


def process_file(path, dry_run=False):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
            # attempt to parse
            content = json.loads(raw)
    except Exception as e:
        return {"file": path, "status": "error", "reason": f"json_load_error: {e}"}

    normalized = normalize_content(content)
    if normalized is None:
        return {"file": path, "status": "skipped", "reason": "unable_to_normalize_top_level_structure"}

    # if normalized is same as content (i.e., top-level list), still we will overwrite to ensure formatting
    # create backup
    backup = make_backup(path)
    # write new content (overwrite)
    if not dry_run:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(normalized, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return {"file": path, "status": "error", "reason": f"write_error: {e}", "backup": backup}
    return {"file": path, "status": "processed", "backup": backup, "items": len(normalized)}


def find_json_files(root="."):
    matches = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                matches.append(os.path.join(dirpath, fn))
    return matches


def main(root=".", dry_run=False):
    files = find_json_files(root)
    results = []
    for p in files:
        r = process_file(p, dry_run=dry_run)
        results.append(r)
    # print summary
    processed = [r for r in results if r["status"] == "processed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]
    print(f"Scanned {len(files)} json files under: {os.path.abspath(root)}")
    print(f"Processed: {len(processed)}, Skipped: {len(skipped)}, Errors: {len(errors)}")
    if processed:
        print("\nProcessed files (sample):")
        for r in processed[:20]:
            print(f" - {r['file']}  items={r.get('items')}  backup={r.get('backup')}")
    if skipped:
        print("\nSkipped files (unable to normalize):")
        for r in skipped[:20]:
            print(f" - {r['file']}  reason={r.get('reason')}")
    if errors:
        print("\nFiles with errors:")
        for r in errors[:20]:
            print(f" - {r['file']}  reason={r.get('reason')}")
    return results


if __name__ == "__main__":
    # set dry_run=True to only simulate without overwriting files
    # change root="." to another directory if needed
    res = main(root="/mnt/home/shensicheng/code/SSD/3_stage_log/qwen3/", dry_run=False)
    # Save a small machine-readable report
    try:
        with open("normalize_json_report.json", "w", encoding="utf-8") as repf:
            json.dump(res, repf, ensure_ascii=False, indent=2)
        print("\nReport saved to normalize_json_report.json")
    except Exception as e:
        print(f"\nFailed to save report: {e}")