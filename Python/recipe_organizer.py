"""
Recipe Organizer CLI
--------------------

A simple CLI to manage recipe JSON files.

Features:
- list: list recipe files in a folder
- view: pretty-print a recipe JSON
- validate: basic schema validation for recipe JSON files
- search: search recipes by ingredient, tag or cuisine
- add: create a new recipe skeleton interactively (writes JSON)
- export: combine multiple recipe JSONs into a single file

Usage examples:
  python recipe_organizer.py list --dir data/samples
  python recipe_organizer.py view data/samples/masala_chai.json
  python recipe_organizer.py validate data/samples/masala_chai.json
  python recipe_organizer.py search --ingredient "cardamom" --dir data/samples
  python recipe_organizer.py add --dir data/samples
  python recipe_organizer.py export --dir data/samples --out combined.json

No external packages required (only Python standard lib).
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------------------------
# Schema & Validation Helpers
# ---------------------------

RECIPE_MINIMAL_KEYS = {
    "id", "title", "description", "cuisine",
    "difficulty", "prep_time", "cook_time",
    "servings", "ingredients", "instructions"
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_recipe_schema(data: Dict[str, Any]) -> (bool, List[str]):
    """
    Lightweight validation: checks presence of required keys and basic types.
    Returns (is_valid, messages)
    """
    messages = []
    keys = set(data.keys())
    missing = RECIPE_MINIMAL_KEYS - keys
    if missing:
        messages.append(f"Missing required keys: {', '.join(sorted(missing))}")

    # Type and content checks (best-effort)
    if "ingredients" in data:
        if not isinstance(data["ingredients"], list):
            messages.append("`ingredients` should be a list of objects")
        else:
            for i, ingr in enumerate(data["ingredients"]):
                if not isinstance(ingr, dict):
                    messages.append(f"`ingredients[{i}]` should be an object")
                else:
                    if "item" not in ingr:
                        messages.append(f"`ingredients[{i}]` missing `item`")
    if "instructions" in data and not isinstance(data["instructions"], list):
        messages.append("`instructions` should be a list of strings")

    valid = len(messages) == 0
    return valid, messages


# ---------------------------
# Core functionality
# ---------------------------

def list_recipes(folder: Path) -> List[Path]:
    pattern = "*.json"
    files = sorted([p for p in folder.glob(pattern) if p.is_file()])
    return files


def pretty_print_json(path: Path) -> None:
    data = load_json(path)
    print(json.dumps(data, indent=2, ensure_ascii=False))


def find_recipes_by_ingredient(folder: Path, ingredient: str) -> List[Path]:
    ingredient_lower = ingredient.strip().lower()
    matches = []
    for p in list_recipes(folder):
        try:
            data = load_json(p)
            ingredients = data.get("ingredients", [])
            for ing in ingredients:
                # item or notes
                item = str(ing.get("item", "")).lower()
                notes = str(ing.get("notes", "")).lower()
                if ingredient_lower in item or ingredient_lower in notes:
                    matches.append(p)
                    break
        except Exception:
            continue
    return matches


def find_recipes_by_tag(folder: Path, tag: str) -> List[Path]:
    tag_lower = tag.strip().lower()
    matches = []
    for p in list_recipes(folder):
        try:
            data = load_json(p)
            tags = [t.lower() for t in data.get("tags", [])]
            if tag_lower in tags:
                matches.append(p)
        except Exception:
            continue
    return matches


def find_recipes_by_cuisine(folder: Path, cuisine: str) -> List[Path]:
    cuisine_lower = cuisine.strip().lower()
    matches = []
    for p in list_recipes(folder):
        try:
            data = load_json(p)
            if data.get("cuisine", "").strip().lower() == cuisine_lower:
                matches.append(p)
        except Exception:
            continue
    return matches


def add_recipe_interactive(folder: Path) -> Path:
    """
    Create a new recipe interactively. Minimal guided prompts.
    """
    folder.mkdir(parents=True, exist_ok=True)
    print("Let's create a new recipe JSON. Press ENTER to accept example/default values when shown.\n")

    def ask(prompt: str, default: Optional[str] = None) -> str:
        if default:
            q = f"{prompt} [{default}]: "
        else:
            q = f"{prompt}: "
        ans = input(q).strip()
        return ans if ans else (default or "")

    rid = ask("Unique id (e.g. masala-chai-2023)", "new-recipe-2025")
    title = ask("Title", "Untitled Recipe")
    description = ask("Short description", "Add a short description here")
    cuisine = ask("Cuisine", "General")
    difficulty = ask("Difficulty (easy/medium/hard)", "easy")
    prep_time = int(ask("Prep time (minutes)", "10") or 0)
    cook_time = int(ask("Cook time (minutes)", "10") or 0)
    servings = int(ask("Servings", "2") or 1)

    print("\nEnter ingredients — input format: item | amount | notes (press enter to stop)")
    ingredients = []
    while True:
        line = input("Ingredient (item | amount | notes): ").strip()
        if not line:
            break
        parts = [p.strip() for p in line.split("|")]
        item = parts[0] if parts else ""
        amount = parts[1] if len(parts) > 1 else ""
        notes = parts[2] if len(parts) > 2 else ""
        ingredients.append({"item": item, "amount": amount, "notes": notes})

    print("\nEnter instructions — each step on a new line (press enter on empty line to stop)")
    instructions = []
    while True:
        step = input("Step: ").strip()
        if not step:
            break
        instructions.append(step)

    tags_raw = ask("Comma-separated tags", "beverage,hot drink")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

    nutrition = {}
    if ask("Add simple nutrition details? (y/n)", "n").lower().startswith("y"):
        calories = ask("Calories (number)", "0")
        nutrition["calories"] = int(calories) if calories.isdigit() else calories
        protein = ask("Protein (e.g. '4g')", "0g")
        carbs = ask("Carbs (e.g. '10g')", "0g")
        fat = ask("Fat (e.g. '3g')", "0g")
        nutrition.update({"protein": protein, "carbs": carbs, "fat": fat})

    data = {
        "id": rid,
        "title": title,
        "description": description,
        "cuisine": cuisine,
        "difficulty": difficulty,
        "prep_time": prep_time,
        "cook_time": cook_time,
        "servings": servings,
        "ingredients": ingredients,
        "instructions": instructions,
        "tags": tags,
        "nutrition": nutrition,
        "image_url": "",
        "author": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
        "date_added": "",  # optional: contributor can add
        "notes": ""
    }

    out_path = folder / f"{rid}.json"
    save_json(out_path, data)
    print(f"\nSaved new recipe to: {out_path}")
    return out_path


def export_recipes(folder: Path, out_file: Path) -> Path:
    combined = []
    for p in list_recipes(folder):
        try:
            d = load_json(p)
            combined.append(d)
        except Exception:
            continue
    save_json(out_file, combined)
    return out_file


# ---------------------------
# CLI Implementation
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recipe_organizer", description="Manage recipe JSON samples")
    p.add_argument("--dir", "-d", default="data/samples", help="Directory containing recipe JSON files")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List recipe JSON files in directory")

    v = sub.add_parser("view", help="Print pretty JSON of a recipe")
    v.add_argument("path", help="Path to recipe JSON file")

    vv = sub.add_parser("validate", help="Validate one or more recipe JSON files")
    vv.add_argument("paths", nargs="+", help="Paths to recipe JSON file(s) or directories")

    s = sub.add_parser("search", help="Search recipes by ingredient, tag or cuisine")
    s.add_argument("--ingredient", "-i", help="Ingredient substring to search")
    s.add_argument("--tag", "-t", help="Tag to search")
    s.add_argument("--cuisine", "-c", help="Cuisine to search")

    a = sub.add_parser("add", help="Interactively add a new recipe JSON")
    a.add_argument("--out", "-o", help="Output directory", default=None)

    e = sub.add_parser("export", help="Export all recipe JSON into a combined JSON array file")
    e.add_argument("--out", "-o", required=True, help="Output file path for combined JSON")

    return p


def run_command(args: argparse.Namespace) -> int:
    base_dir = Path(args.dir)
    if args.cmd == "list":
        files = list_recipes(base_dir)
        if not files:
            print(f"No JSON recipes found in {base_dir}")
            return 0
        print(f"Recipes in {base_dir}:")
        for f in files:
            print(" -", f.name)
        return 0

    if args.cmd == "view":
        path = Path(args.path)
        if not path.exists():
            print("File not found:", path)
            return 2
        pretty_print_json(path)
        return 0

    if args.cmd == "validate":
        had_errors = False
        for p in args.paths:
            pp = Path(p)
            if pp.is_dir():
                targets = list_recipes(pp)
            else:
                targets = [pp]
            for t in targets:
                try:
                    data = load_json(t)
                    ok, messages = validate_recipe_schema(data)
                    if ok:
                        print(f"[OK] {t}")
                    else:
                        had_errors = True
                        print(f"[INVALID] {t}")
                        for m in messages:
                            print("   -", m)
                except Exception as ex:
                    had_errors = True
                    print(f"[ERROR] {t} -> {ex}")
        return (1 if had_errors else 0)

    if args.cmd == "search":
        if args.ingredient:
            matches = find_recipes_by_ingredient(base_dir, args.ingredient)
            print(f"Found {len(matches)} recipes with ingredient '{args.ingredient}':")
            for m in matches:
                print(" -", m.name)
            return 0
        if args.tag:
            matches = find_recipes_by_tag(base_dir, args.tag)
            print(f"Found {len(matches)} recipes with tag '{args.tag}':")
            for m in matches:
                print(" -", m.name)
            return 0
        if args.cuisine:
            matches = find_recipes_by_cuisine(base_dir, args.cuisine)
            print(f"Found {len(matches)} recipes with cuisine '{args.cuisine}':")
            for m in matches:
                print(" -", m.name)
            return 0
        print("Please provide --ingredient or --tag or --cuisine")
        return 2

    if args.cmd == "add":
        out = Path(args.out) if args.out else base_dir
        add_recipe_interactive(out)
        return 0

    if args.cmd == "export":
        out_path = Path(args.out)
        export_recipes(base_dir, out_path)
        print(f"Wrote combined file to {out_path}")
        return 0

    print("Unknown command")
    return 2


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        rc = run_command(args)
        exit(rc)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        exit(2)


if __name__ == "__main__":
    main()
