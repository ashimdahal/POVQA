import json
import argparse
import pandas as pd
from pathlib import Path

def generate_latex_table(summary_files: list, caption: str, label: str) -> str:
    """
    Reads a list of summary.json files, finds the best score for each metric,
    and generates a full-fledged LaTeX table with the highest values bolded.

    Args:
        summary_files: A list of paths to the summary JSON files.
        caption: The caption for the LaTeX table.
        label: The label for referencing the table in LaTeX.

    Returns:
        A string containing the complete LaTeX code for the results table.

    Usage Example (from command line):
        python generate_latex_table.py --run_dir /path/to/project/runs/
    """
    results = []
    for file_path in summary_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Flatten the nested dictionary for easier processing
            method_results = {
                'Method': data.get('method', 'N/A').replace('_', ' ').title()
            }
            metrics = data.get('metrics', {})
            # Format metric names for the table header
            for key, value in metrics.items():
                header_key = key.replace('_', ' ').replace('Bps', ' BP').replace('Cos', ' Cosine').upper()
                method_results[header_key] = value
            results.append(method_results)

    if not results:
        return "No summary files found or files were empty."

    # Use pandas for easy data manipulation and finding max values
    df = pd.DataFrame(results)
    df = df.set_index('Method')

    # Remove the 'EM' column if it exists
    if 'EM' in df.columns:
        df = df.drop(columns=['EM'])

    # Find the maximum value in each remaining metric column
    max_values = df.max()

    # --- Build the LaTeX String ---
    
    # Define the table structure (l for method, c for each metric)
    num_metrics = len(df.columns)
    column_format = 'l' + 'c' * num_metrics
    
    # Start LaTeX table environment
    latex_parts = [
        "\\begin{table*}[ht]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{" + column_format + "}",
        "\\toprule"
    ]

    # Create multi-line headers for better formatting
    header_map = {
        'BLEU4 BP': '\\shortstack{BLEU4\\\\BP}',
        'ROUGE L': '\\shortstack{ROUGE\\\\L}',
        'EMBED COSINE': '\\shortstack{EMBED\\\\COSINE}',
        'ROUGE L REASONING': '\\shortstack{ROUGE L\\\\REASONING}',
        'EMBED COSINE REASONING': '\\shortstack{EMBED COSINE\\\\REASONING}'
    }
    formatted_headers = [header_map.get(col, col) for col in df.columns]
    headers = " & ".join(formatted_headers)
    latex_parts.append(f"\\textbf{{Method}} & {headers} \\\\")
    latex_parts.append("\\midrule")

    # Data Rows
    for method, row in df.iterrows():
        row_values = [method]
        for metric, value in row.items():
            # Format to 3 decimal places
            formatted_value = f"{value:.3f}"
            # Bold the highest value in the column
            if value >= max_values[metric]:
                row_values.append(f"\\textbf{{{formatted_value}}}")
            else:
                row_values.append(formatted_value)
        latex_parts.append(" & ".join(row_values) + " \\\\")

    # End LaTeX table environment
    latex_parts.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])

    return "\n".join(latex_parts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX results table from MovieVQA summary.json files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help="Path to the directory containing your 'runs' and their summary.json files."
    )
    parser.add_argument(
        '--caption',
        type=str,
        default="Main experimental results on our MovieVQA dataset. We report F1, BLEU, ROUGE-L, and Embedding Cosine Similarity for both the final answer and the generated reasoning. The highest score for each metric is highlighted in bold.",
        help="The caption for the generated LaTeX table."
    )
    parser.add_argument(
        '--label',
        type=str,
        default="tab:main_results",
        help="The label for referencing the table in your LaTeX document (e.g., \\ref{tab:main_results})."
    )

    args = parser.parse_args()

    run_directory = Path(args.run_dir)
    # Automatically find all *.summary.json files in the specified directory
    summary_files = list(run_directory.glob('*.summary.json'))

    if not summary_files:
        print(f"Error: No '*.summary.json' files were found in the directory: {args.run_dir}")
    else:
        print(f"Found {len(summary_files)} summary files to process.")
        latex_code = generate_latex_table(summary_files, args.caption, args.label)
        print("\n" + "="*80)
        print("COPY AND PASTE THE LATEX CODE BELOW INTO YOUR .TEX FILE")
        print("="*80 + "\n")
        print(latex_code)
        print("\n" + "="*80)
        print("Requires LaTeX packages: \\usepackage{booktabs}, \\usepackage{caption}, \\usepackage{graphicx} (for table*), and \\usepackage{amsmath} (for shortstack)")
        print("="*80)

