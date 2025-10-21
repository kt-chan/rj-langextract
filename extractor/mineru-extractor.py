import json
import os
import time
from dotenv import load_dotenv
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode


def process_pdf(
    file_path: str,
):
    output_dir = "data/outputs"
    server_url = os.getenv("MINERU_HOST", "http://localhost:30000"),
    f_make_md_mode = MakeMode.MM_MD
    f_dump_md = True
    f_dump_content_list = True
    f_dump_middle_json = True
    f_dump_model_output = True

    start = time.time()
    parts = os.path.splitext(os.path.basename(file_path))
    pdf_file_name = parts[0][:20]
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, 0, None)
    local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, "auto")
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )
    end1 = time.time()
    print(f"start to call sglang, cost, {end1 - start}")
    middle_json, infer_result = vlm_doc_analyze(
        pdf_bytes,
        image_writer=image_writer,
        backend="sglang-client",
        server_url=server_url,
    )
    end2 = time.time()
    print(f"end to call sglang, cost, {end2 - end1}")

    pdf_info = middle_json["pdf_info"]

    # draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
    # draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_md:
        image_dir = str(os.path.basename(local_image_dir))
        md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )
        end3 = time.time()
        print(f"end to gen md, cost, {end3 - end2}")

    if f_dump_content_list:
        image_dir = str(os.path.basename(local_image_dir))
        content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
        md_writer.write_string(
            f"{pdf_file_name}_model_output.txt",
            model_output,
        )

    print(f"local output dir is {local_md_dir}")


if __name__ == "__main__":
    load_dotenv()
    file = r"data/demo/A Study on the Chemical Compositions of the Yinqiaosan (Lonicerae and Forsythiae Powder) at Different Time of Later-decoction by Gas Chromatograph.pdf"
    process_pdf(file)
