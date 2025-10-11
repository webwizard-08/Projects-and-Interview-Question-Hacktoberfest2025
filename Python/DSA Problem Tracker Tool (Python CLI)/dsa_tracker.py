#!/usr/bin/env python3
"""
dsa_tracker.py - Simple DSA Problem Tracker CLI

Usage examples:
  python dsa_tracker.py add --title "Two Sum" --source "LeetCode" --difficulty "Easy" --tags array,hash
  python dsa_tracker.py list --status unsolved --topic array
  python dsa_tracker.py show 1
  python dsa_tracker.py mark 1 solved --notes "Used hash map" --solved-at "2025-10-07"
  python dsa_tracker.py stats
  python dsa_tracker.py export problems_export.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import datetime
from typing import List, Dict, Optional

DATA_PATH = os.path.expanduser("~/.dsa_tracker.json")

def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def load_data() -> List[Dict]:
    if not os.path.exists(DATA_PATH):
        return []
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data
    except Exception as e:
        print(f"Failed to read data file: {e}", file=sys.stderr)
        return []

def save_data(data: List[Dict]):
    try:
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write data file: {e}", file=sys.stderr)

def next_id(data: List[Dict]) -> int:
    if not data:
        return 1
    return max(item.get("id", 0) for item in data) + 1

def parse_tags(tag_str: Optional[str]) -> List[str]:
    if not tag_str:
        return []
    return [t.strip() for t in tag_str.split(",") if t.strip()]

def cmd_add(args):
    data = load_data()
    item = {
        "id": next_id(data),
        "title": args.title,
        "source": args.source or "",
        "url": args.url or "",
        "topic": args.topic or "",
        "difficulty": args.difficulty or "",
        "status": "unsolved",
        "notes": args.notes or "",
        "attempts": 0,
        "tags": parse_tags(args.tags),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "solved_at": None,
    }
    data.append(item)
    save_data(data)
    print(f"Added problem {item['id']}: {item['title']}")

def filter_data(data: List[Dict], args) -> List[Dict]:
    out = data
    if getattr(args, "status", None):
        out = [d for d in out if d.get("status") == args.status]
    if getattr(args, "topic", None):
        out = [d for d in out if args.topic.lower() in (d.get("topic","").lower())]
    if getattr(args, "difficulty", None):
        out = [d for d in out if d.get("difficulty","").lower() == args.difficulty.lower()]
    if getattr(args, "tag", None):
        out = [d for d in out if args.tag.lower() in [t.lower() for t in d.get("tags",[])]]
    return out

def format_item_brief(item):
    tags = ",".join(item.get("tags",[]))
    return f"{item['id']:3d} | {item['title'][:45]:45s} | {item.get('difficulty',''):7s} | {item.get('status',''):8s} | {tags}"

def cmd_list(args):
    data = load_data()
    data = sorted(data, key=lambda x: (x.get("status") != "unsolved", x.get("id")))
    data = filter_data(data, args)
    if not data:
        print("No problems match.")
        return
    print("ID  | Title                                           | Diff    | Status   | Tags")
    print("-"*90)
    for item in data:
        print(format_item_brief(item))

def find_by_id(data, id_):
    for d in data:
        if d.get("id") == id_:
            return d
    return None

def cmd_show(args):
    data = load_data()
    item = find_by_id(data, args.id)
    if not item:
        print("Not found.")
        return
    # Pretty print details
    print("="*60)
    print(f"ID:       {item['id']}")
    print(f"Title:    {item['title']}")
    print(f"Source:   {item.get('source') or '-'}")
    print(f"URL:      {item.get('url') or '-'}")
    print(f"Topic:    {item.get('topic') or '-'}")
    print(f"Difficulty:{item.get('difficulty') or '-'}")
    print(f"Status:   {item.get('status')}")
    print(f"Attempts: {item.get('attempts')}")
    print(f"Tags:     {', '.join(item.get('tags',[])) or '-'}")
    print(f"Created:  {item.get('created_at')}")
    print(f"Updated:  {item.get('updated_at')}")
    print(f"Solved at:{item.get('solved_at') or '-'}")
    print("-"*60)
    print("Notes:")
    print(item.get("notes") or "-")
    print("="*60)

def cmd_update(args):
    data = load_data()
    item = find_by_id(data, args.id)
    if not item:
        print("Not found.")
        return
    changed = False
    if args.title:
        item["title"] = args.title; changed = True
    if args.source:
        item["source"] = args.source; changed = True
    if args.url:
        item["url"] = args.url; changed = True
    if args.topic:
        item["topic"] = args.topic; changed = True
    if args.difficulty:
        item["difficulty"] = args.difficulty; changed = True
    if args.notes:
        item["notes"] = args.notes; changed = True
    if args.tags is not None:
        item["tags"] = parse_tags(args.tags); changed = True
    if changed:
        item["updated_at"] = now_iso()
        save_data(data)
        print("Updated.")
    else:
        print("No changes provided.")

def cmd_mark(args):
    data = load_data()
    item = find_by_id(data, args.id)
    if not item:
        print("Not found.")
        return
    new_status = args.status.lower()
    if new_status not in ("unsolved","attempted","solved"):
        print("Status must be one of: unsolved, attempted, solved")
        return
    item["status"] = new_status
    if args.notes:
        item["notes"] = (item.get("notes","") + "\n" + args.notes).strip()
    if args.increment_attempts:
        try:
            item["attempts"] = int(item.get("attempts",0)) + int(args.increment_attempts)
        except:
            item["attempts"] = item.get("attempts",0) + 1
    if new_status == "solved":
        item["solved_at"] = args.solved_at or now_iso()
    item["updated_at"] = now_iso()
    save_data(data)
    print(f"Marked {item['id']} as {new_status}.")

def cmd_delete(args):
    data = load_data()
    item = find_by_id(data, args.id)
    if not item:
        print("Not found.")
        return
    data = [d for d in data if d.get("id") != args.id]
    save_data(data)
    print(f"Deleted {args.id}.")

def cmd_search(args):
    data = load_data()
    q = args.query.lower()
    results = [d for d in data if q in d.get("title","").lower() or q in (d.get("notes") or "").lower() or q in d.get("source","").lower()]
    if not results:
        print("No matches.")
        return
    results = sorted(results, key=lambda x: x.get("id"))
    print("ID  | Title                                           | Diff    | Status   | Tags")
    print("-"*90)
    for item in results:
        print(format_item_brief(item))

def cmd_stats(args):
    data = load_data()
    total = len(data)
    solved = sum(1 for d in data if d.get("status")=="solved")
    attempted = sum(1 for d in data if d.get("status")=="attempted")
    unsolved = sum(1 for d in data if d.get("status")=="unsolved")
    by_topic = {}
    by_difficulty = {}
    for d in data:
        t = d.get("topic") or "Unknown"
        by_topic[t] = by_topic.get(t,0)+1
        diff = d.get("difficulty") or "Unknown"
        by_difficulty[diff] = by_difficulty.get(diff,0)+1
    print("Problems total:", total)
    print("Solved:", solved)
    print("Attempted:", attempted)
    print("Unsolved:", unsolved)
    print()
    print("By difficulty:")
    for k,v in sorted(by_difficulty.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k:10s} : {v}")
    print()
    print("Top topics:")
    for k,v in sorted(by_topic.items(), key=lambda x: (-x[1], x[0]))[:10]:
        print(f"  {k:15s} : {v}")

def cmd_export(args):
    data = load_data()
    out_path = args.path
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(data)} items to {out_path}")
    except Exception as e:
        print(f"Failed to export: {e}")

def cmd_import(args):
    data = load_data()
    try:
        with open(args.path, "r", encoding="utf-8") as f:
            new = json.load(f)
        if not isinstance(new, list):
            print("Import file must be a JSON list of problem objects.")
            return
        # Merge: preserve ids by offsetting if necessary
        existing_ids = {d.get("id") for d in data}
        start_id = next_id(data)
        added = 0
        for item in new:
            if not isinstance(item, dict):
                continue
            if item.get("id") in existing_ids or item.get("id") is None:
                item["id"] = start_id
                start_id += 1
            if "created_at" not in item:
                item["created_at"] = now_iso()
            item["updated_at"] = item.get("updated_at") or now_iso()
            data.append(item)
            added += 1
        save_data(data)
        print(f"Imported {added} items.")
    except Exception as e:
        print(f"Import failed: {e}")

def build_parser():
    p = argparse.ArgumentParser(prog="dsa_tracker", description="DSA Problem Tracker CLI")
    sub = p.add_subparsers(dest="cmd")

    # add
    pa = sub.add_parser("add", help="Add a new problem")
    pa.add_argument("--title", required=True)
    pa.add_argument("--source", help="Where it's from (LeetCode, CF, Book)")
    pa.add_argument("--url", help="Optional URL")
    pa.add_argument("--topic", help="Topic (arrays, dp)")
    pa.add_argument("--difficulty", help="Easy/Medium/Hard")
    pa.add_argument("--tags", help="Comma-separated tags")
    pa.add_argument("--notes", help="Notes")
    pa.set_defaults(func=cmd_add)

    # list
    pl = sub.add_parser("list", help="List problems")
    pl.add_argument("--status", choices=["unsolved","attempted","solved"], help="Filter by status")
    pl.add_argument("--topic", help="Filter by topic substring")
    pl.add_argument("--difficulty", help="Filter by difficulty")
    pl.add_argument("--tag", help="Filter by tag")
    pl.set_defaults(func=cmd_list)

    # show
    ps = sub.add_parser("show", help="Show details for a problem by ID")
    ps.add_argument("id", type=int)
    ps.set_defaults(func=cmd_show)

    # update
    pu = sub.add_parser("update", help="Update fields for a problem")
    pu.add_argument("id", type=int)
    pu.add_argument("--title")
    pu.add_argument("--source")
    pu.add_argument("--url")
    pu.add_argument("--topic")
    pu.add_argument("--difficulty")
    pu.add_argument("--notes")
    pu.add_argument("--tags", help="Set tags (comma-separated). Pass empty string to clear.")
    pu.set_defaults(func=cmd_update)

    # mark
    pm = sub.add_parser("mark", help="Change status (unsolved/attempted/solved) and optionally add notes")
    pm.add_argument("id", type=int)
    pm.add_argument("status", choices=["unsolved","attempted","solved"])
    pm.add_argument("--notes")
    pm.add_argument("--increment-attempts", dest="increment_attempts", help="Number or blank to add 1")
    pm.add_argument("--solved-at", help="ISO datetime string for solved time")
    pm.set_defaults(func=cmd_mark)

    # delete
    pd = sub.add_parser("delete", help="Delete a problem")
    pd.add_argument("id", type=int)
    pd.set_defaults(func=cmd_delete)

    # search
    psr = sub.add_parser("search", help="Full text search (title, notes, source)")
    psr.add_argument("query")
    psr.set_defaults(func=cmd_search)

    # stats
    pst = sub.add_parser("stats", help="Show summary statistics")
    pst.set_defaults(func=cmd_stats)

    # export
    pex = sub.add_parser("export", help="Export DB to JSON file")
    pex.add_argument("path")
    pex.set_defaults(func=cmd_export)

    # import
    pim = sub.add_parser("import", help="Import JSON file")
    pim.add_argument("path")
    pim.set_defaults(func=cmd_import)

    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
