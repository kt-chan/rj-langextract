import langextract as lx
import textwrap
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor"}
            ),
        ]
    )
]

if __name__ == "__main__":
    # The input text to be processed
    input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

    # Run the extraction
    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id="gpt-4o",
        api_key=os.getenv("LANGEXTRACT_API_KEY"),  # Load from environment variable
        model_url=os.getenv("LANGEXTRACT_MODEL_URL", "https://api.laozhang.ai/v1"),  # Load from environment variable with fallback
    )

    # Save the results to a JSONL file
    lx.io.save_annotated_documents([result], output_name="data/extraction_results.jsonl", output_dir=".")

    # Generate the visualization from the file
    html_content = lx.visualize("data/extraction_results.jsonl")
    with open("data/visualization.html", "w", encoding="utf-8") as f:
        if hasattr(html_content, 'data'):
            f.write(html_content.data)  # For Jupyter/Colab
        else:
            f.write(html_content)