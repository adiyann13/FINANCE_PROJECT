from pathlib import Path

import pymupdf4llm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "markdown"


def main() -> None:
    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_files:
        markdown = pymupdf4llm.to_markdown(str(pdf_path))
        output_path = OUTPUT_DIR / f"{pdf_path.stem}.md"
        output_path.write_text(markdown, encoding="utf-8")
        print(f"Converted {pdf_path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()