import re
import base64
import io
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf
from PIL import Image
from ..config import settings

class FinancialDocumentParser:
    """
    Multimodal parser for financial disclosures, PDFs, markdown, and chart images.
    Extracts both textual passages and visual chart descriptions (via Vision Agent).
    """

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None, vision_agent: Optional[Any] = None):
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        if self.chunk_size <= 0 or self.chunk_overlap < 0:
            raise ValueError("chunk_size must be positive and chunk_overlap cannot be negative.")
        if vision_agent is None:
            from ..agents.vision_agent import VisionAgent
            self.vision_agent = VisionAgent()
        else:
            self.vision_agent = vision_agent

    def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parses a document or image file into text or visual chunks."""
        file_path = Path(file_path)
        if not file_path.exists():
            return []

        ext = file_path.suffix.lower()

        if ext in [".png", ".jpg", ".jpeg", ".webp", ".svg"]:
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

        for file_path in sorted(dir_path.glob("*.*")):
            if file_path.suffix.lower() in [".md", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".svg"]:
                chunks = self.parse_file(file_path)
                all_chunks.extend(chunks)

        return all_chunks

    def _split_coherent_text(self, text: str) -> List[str]:
        """Split at sentence boundaries and preserve the configured overlap.

        PDF extraction often provides one long, line-wrapped paragraph. The
        former sliding word window cut thoughts mid-sentence and hard-coded a
        20-word overlap, ignoring the configured value.
        """
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            sentences = [text]

        units: List[str] = []
        for sentence in sentences:
            if len(sentence) <= self.chunk_size:
                units.append(sentence)
                continue
            words = sentence.split()
            current: List[str] = []
            current_len = 0
            for word in words:
                if current and current_len + len(word) + 1 > self.chunk_size:
                    units.append(" ".join(current))
                    current = []
                    current_len = 0
                current.append(word)
                current_len += len(word) + (1 if current_len else 0)
            if current:
                units.append(" ".join(current))

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for unit in units:
            if current and current_len + len(unit) + 1 > self.chunk_size:
                chunks.append(" ".join(current))
                overlap: List[str] = []
                overlap_len = 0
                for prior in reversed(current):
                    added = len(prior) + (1 if overlap else 0)
                    if overlap and overlap_len + added > self.chunk_overlap:
                        break
                    overlap.insert(0, prior)
                    overlap_len += added
                current = overlap
                current_len = sum(len(part) for part in current) + max(0, len(current) - 1)
            current.append(unit)
            current_len += len(unit) + (1 if current_len else 0)
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _parse_standalone_image(self, file_path: Path) -> List[Dict[str, Any]]:
        """Uses Vision Agent to describe image/chart and creates a visual chunk."""
        filename = file_path.name
        doc_id = file_path.stem.lower()
        
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
        doc_id = file_path.stem.lower()

        try:
            reader = pypdf.PdfReader(str(file_path))
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue

                # Normalize text line breaks
                normalized_text = page_text.replace("\r\n", "\n").replace("\r", "\n")
                
                # Split by double newlines or major structural divisions
                raw_paras = [p.strip() for p in re.split(r'\n\s*\n', normalized_text) if p.strip()]
                if not raw_paras:
                    raw_paras = [normalized_text.strip()]

                page_chunks = []
                for para in raw_paras:
                    # Clean single line wrap breaks within paragraph
                    clean_para = re.sub(r'(?<!\n)\n(?!\n)', ' ', para).strip()
                    if len(clean_para) <= self.chunk_size:
                        if len(clean_para) > 15:
                            page_chunks.append(clean_para)
                    else:
                        page_chunks.extend(self._split_coherent_text(clean_para))

                for p_idx, p_text in enumerate(page_chunks):
                    chunk_id = f"{doc_id}_p{page_num}_c{p_idx+1}"
                    # Try to infer heading from first line or words
                    first_words = " ".join(p_text.split()[:5])
                    chunks.append({
                        "id": chunk_id,
                        "doc_id": doc_id,
                        "text": p_text,
                        "source_document": filename,
                        "page_number": page_num,
                        "section_title": f"Page {page_num}: {first_words}...",
                        "chunk_type": "text",
                        "image_base64": None
                    })

                # Embedded images extraction from PDF
                if hasattr(page, 'images'):
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
                            pass

        except Exception as e:
            print(f"Error reading PDF {filename}: {e}")

        return chunks

    def _parse_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parses Markdown/text documents."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        filename = file_path.name
        doc_id = file_path.stem.lower()

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

            for chunk_text in self._split_coherent_text(clean_sec):
                if len(chunk_text) <= 10:
                    continue
                chunk_id = f"{doc_id}_p{current_page}_c{len(chunks)+1}"
                chunks.append({
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "source_document": filename,
                    "page_number": current_page,
                    "section_title": current_section_title,
                    "chunk_type": "text",
                    "image_base64": None
                })

        return chunks
