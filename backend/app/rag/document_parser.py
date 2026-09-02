import re
import base64
import io
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf
from PIL import Image
from ..agents.vision_agent import VisionAgent

class FinancialDocumentParser:
    """
    Multimodal parser for financial disclosures, PDFs, markdown, and chart images.
    Extracts both textual passages and visual chart descriptions (via Vision Agent).
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100, vision_agent: Optional[VisionAgent] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vision_agent = vision_agent or VisionAgent()

    def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parses a document or image file into text or visual chunks."""
        file_path = Path(file_path)
        if not file_path.exists():
            return []

        ext = file_path.suffix.lower()

        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
            return self._parse_standalone_image(file_path)
        elif ext == ".svg":
            return self._parse_standalone_image(file_path)
        elif ext == ".pdf":
            return self._parse_pdf(file_path)
        elif ext in [".md", ".txt"]:
            return self._parse_text_file(file_path)
        else:
            return self._parse_text_file(file_path)

    def parse_directory(self, dir_path: Path) -> List[Dict[str, Any]]:
        """Parses all supported files in a directory."""
        dir_path = Path(dir_path)
        all_chunks: List[Dict[str, Any]] = []
        if not dir_path.exists():
            return all_chunks

        # Sort so text files are indexed first, then charts
        for file_path in sorted(dir_path.glob("*.*")):
            if file_path.suffix.lower() in [".md", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".svg"]:
                chunks = self.parse_file(file_path)
                all_chunks.extend(chunks)

        return all_chunks

    def _parse_standalone_image(self, file_path: Path) -> List[Dict[str, Any]]:
        """Uses Vision Agent to describe image/chart and creates a visual chunk."""
        filename = file_path.name
        doc_id = filename.split(".")[0].lower()
        
        visual_data = self.vision_agent.describe_image(
            image_input=file_path,
            filename=filename,
            page_number=1
        )
        
        chunk_id = f"{doc_id}_img_p1_c1"
        return [{
            "id": chunk_id,
            "doc_id": doc_id,
            "text": visual_data["text"],
            "source_document": filename,
            "page_number": 1,
            "section_title": visual_data["section_title"],
            "chunk_type": "visual",
            "image_base64": visual_data.get("image_base64")
        }]

    def _parse_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extracts text and embedded images from a PDF file."""
        chunks: List[Dict[str, Any]] = []
        filename = file_path.name
        doc_id = filename.split(".")[0].lower()

        try:
            reader = pypdf.PdfReader(str(file_path))
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                page_text = page.extract_text() or ""

                # Text chunks from page
                paras = [p.strip() for p in page_text.split("\n\n") if len(p.strip()) > 20]
                for p_idx, para in enumerate(paras):
                    chunk_id = f"{doc_id}_p{page_num}_c{p_idx+1}"
                    chunks.append({
                        "id": chunk_id,
                        "doc_id": doc_id,
                        "text": para,
                        "source_document": filename,
                        "page_number": page_num,
                        "section_title": f"Page {page_num} Passage",
                        "chunk_type": "text",
                        "image_base64": None
                    })

                # Embedded images extraction from PDF
                for img_idx, img_obj in enumerate(page.images):
                    try:
                        img_bytes = img_obj.data
                        visual_data = self.vision_agent.describe_image(
                            image_input=img_bytes,
                            filename=f"{filename}_{img_obj.name}",
                            page_number=page_num
                        )
                        chunk_id = f"{doc_id}_p{page_num}_img{img_idx+1}"
                        chunks.append({
                            "id": chunk_id,
                            "doc_id": doc_id,
                            "text": visual_data["text"],
                            "source_document": filename,
                            "page_number": page_num,
                            "section_title": f"Figure on Page {page_num}: {img_obj.name}",
                            "chunk_type": "visual",
                            "image_base64": visual_data.get("image_base64")
                        })
                    except Exception as img_err:
                        print(f"Warning: Failed to extract embedded image {img_obj.name} from {filename}: {img_err}")

        except Exception as e:
            print(f"Error reading PDF {filename}: {e}")

        return chunks

    def _parse_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parses Markdown/text documents."""
        content = file_path.read_text(encoding="utf-8")
        filename = file_path.name
        doc_id = filename.split(".")[0].lower()

        raw_sections = re.split(r'\n(?=#{1,3}\s|---|===)', content)
        chunks: List[Dict[str, Any]] = []
        
        current_page = 1
        current_section_title = "Overview"

        for sec_idx, section in enumerate(raw_sections):
            clean_sec = section.strip()
            if not clean_sec or clean_sec in ["---", "==="]:
                current_page += 1
                continue

            header_match = re.match(r'^(#{1,3})\s+(.+)', clean_sec)
            if header_match:
                current_section_title = header_match.group(2).strip()

            if len(clean_sec) <= self.chunk_size:
                chunk_id = f"{doc_id}_p{current_page}_c{len(chunks)+1}"
                chunks.append({
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "text": clean_sec,
                    "source_document": filename,
                    "page_number": current_page,
                    "section_title": current_section_title,
                    "chunk_type": "text",
                    "image_base64": None
                })
            else:
                paras = [p.strip() for p in clean_sec.split("\n\n") if p.strip()]
                for p_idx, para in enumerate(paras):
                    if len(para) < 20:
                        continue
                    chunk_id = f"{doc_id}_p{current_page}_c{len(chunks)+1}"
                    chunks.append({
                        "id": chunk_id,
                        "doc_id": doc_id,
                        "text": para,
                        "source_document": filename,
                        "page_number": current_page,
                        "section_title": current_section_title,
                        "chunk_type": "text",
                        "image_base64": None
                    })

            if sec_idx > 0 and sec_idx % 2 == 0:
                current_page += 1

        return chunks
