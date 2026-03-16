#!/bin/bash
# Fix parquet files that are actually ZIP files (Kaggle quirk)
# Usage: bash fix_zipped_parquets.sh <base_dir> [<base_dir2> ...]

fix_dir() {
    local dir="$1"
    local fixed=0 skipped=0 errors=0

    echo "Processing directory: $dir"
    echo "============================================="

    while IFS= read -r -d '' f; do
        magic=$(head -c 4 "$f" | xxd -p)

        if [ "$magic" != "504b0304" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        echo "Fixing zipped parquet: $f"

        tmpdir=$(mktemp -d)
        if unzip -o -q "$f" -d "$tmpdir" 2>/dev/null; then
            # Find the extracted parquet file
            extracted=$(find "$tmpdir" -name "*.parquet" -type f | head -1)

            if [ -n "$extracted" ]; then
                size=$(du -h "$extracted" | cut -f1)
                mv "$extracted" "$f"
                echo "  Extracted -> $f ($size)"
                fixed=$((fixed + 1))
            else
                echo "  WARNING: No .parquet found after extraction"
                errors=$((errors + 1))
            fi
        else
            echo "  FAILED to unzip $f"
            errors=$((errors + 1))
        fi
        rm -rf "$tmpdir"

    done < <(find "$dir" -name "*.parquet" -print0 | sort -z)

    echo "============================================="
    echo "Done. Fixed: $fixed, Already OK: $skipped, Errors: $errors"
    echo ""
}

if [ $# -eq 0 ]; then
    echo "Usage: bash fix_zipped_parquets.sh <dir1> [<dir2> ...]"
    exit 1
fi

for dir in "$@"; do
    fix_dir "$dir"
done
