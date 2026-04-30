#!/bin/bash
# Quick runner script for FMNV evaluations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYPLOM_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 FMNV Evaluation Runner"
echo "========================"
echo ""
echo "Location: $SCRIPT_DIR"
echo "Dyplom: $DYPLOM_DIR"
echo ""

if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY not set"
    echo ""
    echo "Set it with:"
    echo "  export GEMINI_API_KEY=\"your-key-here\""
    exit 1
fi

# Parse arguments
CONFIG="${1:-F}"

case "$CONFIG" in
    F)
        echo "Running Config F (3.1-flash-lite) — Quick test"
        cd "$DYPLOM_DIR"
        python fmnv_eval/fmnv_pipeline_eval.py --config F
        ;;
    all)
        echo "Running all Configs A–G"
        cd "$DYPLOM_DIR"
        python fmnv_eval/fmnv_pipeline_eval.py --all
        ;;
    compare)
        echo "Comparing results"
        cd "$DYPLOM_DIR"
        python fmnv_eval/fmnv_pipeline_eval.py --compare
        ;;
    A|B|C|D|E|G)
        echo "Running Config $CONFIG"
        cd "$DYPLOM_DIR"
        python fmnv_eval/fmnv_pipeline_eval.py --config "$CONFIG"
        ;;
    H)
        echo "Running Config H (Gemma-4 local)"
        cd "$DYPLOM_DIR"
        python fmnv_eval/fmnv_gemma_eval.py
        ;;
    *)
        echo "Usage: $0 [F|A|B|C|D|E|G|all|compare|H]"
        echo ""
        echo "Examples:"
        echo "  $0 F       — Run Config F (recommended, quick test)"
        echo "  $0 all     — Run all Gemini configs A–G"
        echo "  $0 compare — Print results comparison table"
        echo "  $0 H       — Run Config H (Gemma-4, PC only)"
        echo ""
        echo "Set API key first:"
        echo "  export GEMINI_API_KEY=\"your-key-here\""
        exit 1
        ;;
esac

echo ""
echo "✅ Done!"
