import os
from readline import redisplay
import time
import json
import shutil
from IPython.display import display
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from collections import Counter

def extract_pdf_content(pdf_path, output_dir="data/output"):
    """
    Extract text and tables from a PDF document with proper table formatting

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract elements from PDF with table structure preservation
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",  # Use high resolution strategy for table extraction
        chunking_strategy="by_title",
        infer_table_structure=True,  # Crucial for table extraction
        # model_name="yolox",  # Best model for table extraction :cite[2]
        # languages=["eng"],  # Specify language for OCR
        skip_infer_table_types=False,  # Ensure tables are processed
    )

    display(Counter(type(element) for element in elements))
    
    # Save all elements to JSON for reference
    base_name = os.path.basename(pdf_path).rsplit(".", 1)[0][:20]
    json_path = os.path.join(output_dir, f"{base_name}_elements.json")
    elements_to_json(elements, filename=json_path)

    # Process elements to separate text and tables
    text_content = []
    current_title = []
    tables = []

    for element in elements:
        if element.category == "Title":
            # Store the title as a potential table header
            current_title = element.text
        if element.category == "Table":
            # Extract table as HTML for proper formatting
            table_html = element.metadata.text_as_html
            tables.append(
                {
                    "text": element.text,
                    "html": table_html,
                    "page_number": element.metadata.page_number,
                    "header": current_title,
                }
            )
        else:
            text_content.append(
                {
                    "text": element.text,
                    "type": element.category,
                    "page_number": element.metadata.page_number,
                }
            )
        # Reset title after associating with a table
        current_title = None

    # Save text content to file
    text_path = os.path.join(output_dir, f"{base_name}_text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for item in text_content:
            f.write(f"[{item['type']} - Page {item['page_number']}]\n")
            f.write(f"{item['text']}\n\n")

    # Save tables to HTML files
    for i, table in enumerate(tables):
        table_path = os.path.join(output_dir, f"{base_name}_table_{i+1}.html")

        # Extract header information if available
        header_html = ""
        if "header" in table and table["header"]:
            # Create header row from the title/text
            header_html = f"""
            <tr>
                <th colspan="100%" style="background-color: #e0e0e0; font-weight: bold; text-align: center;">
                    {table['header']}
                </th>
            </tr>
            """

        # Insert header into the table structure
        table_html = table["html"]
        if header_html and "<thead>" in table_html:
            # Insert header row into the existing thead section
            table_html = table_html.replace("<thead>", f"<thead>{header_html}")
        elif header_html:
            # Create a new thead section with the header
            table_html = table_html.replace(
                "<table>", f"<table><thead>{header_html}</thead>"
            )

        # Create styled HTML document for better table rendering
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Table {i+1} - Page {table['page_number']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .page-info {{ color: #666; font-style: italic; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="page-info">Page {table['page_number']}</div>
            {table_html}
        </body>
        </html>
        """

        with open(table_path, "w", encoding="utf-8") as f:
            f.write(styled_html)

    # Save tables metadata to JSON
    tables_path = os.path.join(output_dir, f"{base_name}_tables.json")
    with open(tables_path, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(text_content)} text elements and {len(tables)} tables")
    print(f"Results saved to {output_dir} directory")

    return text_content, tables


def process_pdf_batch(input_dir, output_dir, max_workers=2):
    """
    Process multiple PDFs in parallel for better performance :cite[1]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all PDF files in input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    def process_single_pdf(pdf_file):
        try:
            start_time = time.time()
            pdf_path = os.path.join(input_dir, pdf_file)

            # Create subdirectory for this PDF's output
            pdf_output_dir = os.path.join(
                output_dir, os.path.splitext(pdf_file)[0][:20]
            )
            os.makedirs(pdf_output_dir, exist_ok=True)

            # Process the PDF
            text_elements, tables = extract_pdf_content(pdf_path, pdf_output_dir)

            processing_time = time.time() - start_time
            print(f"Processed {pdf_file} in {processing_time:.2f} seconds")

            return True
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            return False

    # Process in parallel with limited workers to avoid memory issues
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_pdf, pdf_files))

    successful = sum(results)
    print(f"Successfully processed {successful} out of {len(pdf_files)} files")
    return successful


def file_count_under_directory(directory_path: str):
    # Iterate through the directory
    file_count = 0
    for filename in os.listdir(directory_path):
        # Check if the file is a PDF
        if filename.endswith(".pdf"):
            file_count += 1
    return file_count


def truncate_directory(directory_path: str):
    """
    Truncate the specified directory by removing all files and subdirectories.

    :param directory_path: Path to the directory to be truncated
    """
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # Check if the path is indeed a directory
    if not os.path.isdir(directory_path):
        print(f"The path {directory_path} is not a directory.")
        return

    # Iterate through all files and subdirectories in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if it is a file or a directory
        if os.path.isfile(file_path) or os.path.islink(file_path):
            # Remove the file or symbolic link
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            # Remove the subdirectory and all its contents
            shutil.rmtree(file_path)

    print(f"The directory {directory_path} has been truncated.")


# Example usage
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    current_directory = Path(os.getcwd())
    input_pdfs_directory = str(Path(current_directory / "data/demo").resolve())
    output_text_directory = str(Path(current_directory / "data/outputs").resolve())
    truncate_directory(output_text_directory)
    process_pdf_batch(
        input_pdfs_directory,
        output_text_directory,
        max_workers=min(file_count_under_directory(input_pdfs_directory), 4),
    )
