import langextract as lx
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Define the data structure for a chemical compound
class ChemicalCompound(BaseModel):
    """Represents a chemical compound."""
    compound_name: str = Field(..., description="The common name of the chemical compound.")
    compound_cid: Optional[int] = Field(None, description="The Compound ID (CID).")
    molecular_formula: Optional[str] = Field(None, description="The molecular formula (MF).")
    molecular_weight: Optional[str] = Field(None, description="The molecular weight (MW).")
    iupac_name: Optional[str] = Field(None, description="The IUPAC name.")
    smiles: Optional[str] = Field(None, description="The SMILES string.")
    inchl_key: Optional[str] = Field(None, description="The InChIKey.")
    inchl: Optional[str] = Field(None, description="The InChI string.")
    create_date: Optional[str] = Field(None, description="The creation date of the record.")

# Text with a chemical compound mention
source_text = """
# Protocatechuic acid 4-O-glucoside; DTXSID301341731

Compound CID: 157010113

MF: C13H16O9 MW: 316.26 g/mol

IUPAC Name: 3- hydroxy- 4- [3,4,5- trihydroxy- 6- (hydroxymethyl)oxan- 2- yl]oxybenzoic acid

SMILES: C1=C(C=C(C=C1C(=O)O)O)OC2C(C(C(C(O2)CO)O)O)O

InChlKey: HFFREILXLCWCQH- UHFFFAOYSA- N

InChl: InChl=1S/C13H16O9/c14- 4- 8- 9(16)10(17)11(18)13(22- 8)21- 7- 2- 1- 5(12(19)20)3- 6(7)15/h1- 3,8- 11,13- 18H,4H2.(H,19,20)

Create Date: 2021- 11- 15
"""

# Define example data with entities
example_text = """
# 5- fluoro- protocatechuic acid; SCHEMBL9787632

Compound CID: 15173704

MF: C7H5FO4 MW: 172.11 g/mol

IUPAC Name: 3- fluoro- 4,5- dihydroxybenzoic acid

SMILES: C1=C(C=C(C(=C1O)O)F)C(=O)O

InChlKey: RTYDIWLGIZQTKB- UHFFFAOYSA- N

InChl: InChl=1S/C7H5FO4/c8- 4- 1- 3(7(11)12)2- 5(9)6(4)10/h1- 2,9- 10H, (H,11,12)

Create Date: 2007- 02- 09
"""

examples = [
    lx.data.ExampleData(
        text=example_text,
        extractions=[
            lx.data.Extraction(extraction_class="compound_name", extraction_text="5- fluoro- protocatechuic acid"),
            lx.data.Extraction(extraction_class="compound_cid", extraction_text="15173704"),
            lx.data.Extraction(extraction_class="molecular_formula", extraction_text="C7H5FO4"),
            lx.data.Extraction(extraction_class="molecular_weight", extraction_text="172.11 g/mol"),
            lx.data.Extraction(extraction_class="iupac_name", extraction_text="3- fluoro- 4,5- dihydroxybenzoic acid"),
            lx.data.Extraction(extraction_class="smiles", extraction_text="C1=C(C=C(C(=C1O)O)F)C(=O)O"),
            lx.data.Extraction(extraction_class="inchl_key", extraction_text="RTYDIWLGIZQTKB- UHFFFAOYSA- N"),
            lx.data.Extraction(extraction_class="inchl", extraction_text="InChl=1S/C7H5FO4/c8- 4- 1- 3(7(11)12)2- 5(9)6(4)10/h1- 2,9- 10H, (H,11,12)"),
            lx.data.Extraction(extraction_class="create_date", extraction_text="2007-02-09"),
        ]
    )
]

# Extract the information
result = lx.extract(
    text_or_documents=source_text,
    prompt_description="Extract chemical compound information from the text.",
    examples=examples,
    model_id=os.getenv("MODEL_ID", "gpt-4o"),
    api_key=os.getenv("LANGEXTRACT_API_KEY"),
    model_url=os.getenv("LANGEXTRACT_MODEL_URL", "https://api.laozhang.ai/v1"),
)

# Display the extracted information
if result.extractions:
    # When using a schema, the extractions are Pydantic models
    for compound in result.extractions:
        if isinstance(compound, ChemicalCompound):
            print("Extracted Chemical Compound Information:")
            print(f"  - Compound Name: {compound.compound_name}")
            print(f"  - Compound CID: {compound.compound_cid}")
            print(f"  - Molecular Formula: {compound.molecular_formula}")
            print(f"  - Molecular Weight: {compound.molecular_weight}")
            print(f"  - IUPAC Name: {compound.iupac_name}")
            print(f"  - SMILES: {compound.smiles}")
            print(f"  - InChIKey: {compound.inchl_key}")
            print(f"  - InChI: {compound.inchl}")
            print(f"  - Create Date: {compound.create_date}")
else:
    print("No chemical compound information was extracted.")

# Save the results
lx.io.save_annotated_documents([result], output_name="data/chemical_compound_extraction.jsonl", output_dir=".")
# Generate the interactive visualization
html_content = lx.visualize("data/chemical_compound_extraction.jsonl")
with open("data/chemical_compound_extraction.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Interactive visualization saved to chemical_compound_extraction.html")