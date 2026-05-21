# pdf_parser.py
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Image, NarrativeText, Title
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self, image_output_dir: str = "./data/extracted_images"):
        self.image_output_dir = image_output_dir
        Path(image_output_dir).mkdir(parents=True, exist_ok=True)

    def parse(self, pdf_path: str) -> dict:
        logger.info(f"Parsing PDF: {pdf_path}")

        elements = partition_pdf(
            filename=pdf_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            strategy="hi_res",
            hi_res_model_name="detectron2_onnx",
            image_output_dir_path=self.image_output_dir,
        )

        return {
            "text_elements":  [e for e in elements if isinstance(e, (NarrativeText, Title))],
            "table_elements": [e for e in elements if isinstance(e, Table)],
            "image_elements": [e for e in elements if isinstance(e, Image)],
            "source_file":    pdf_path
        }
