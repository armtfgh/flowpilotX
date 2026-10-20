import json
import zipfile
from io import BytesIO

import fitz
from openpyxl import Workbook

from flora_translate.inventory_profiles import extract_document_text


def _zip_bytes(members: dict[str, str]) -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return stream.getvalue()


def test_extracts_supported_text_and_json_documents():
    assert "pump" in extract_document_text("inventory.txt", b"pump list")
    output = extract_document_text("inventory.json", json.dumps({"BPR_available": [3]}).encode())
    assert "BPR_available" in output


def test_extracts_docx_and_pptx_ooxml_without_office_dependencies():
    docx = _zip_bytes(
        {
            "word/document.xml": (
                '<w:document xmlns:w="urn:test"><w:body><w:p><w:r>'
                "<w:t>No inline degasser</w:t></w:r></w:p></w:body></w:document>"
            )
        }
    )
    pptx = _zip_bytes(
        {
            "ppt/slides/slide1.xml": (
                '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>10 mL reactor</a:t></p:sld>'
            )
        }
    )
    assert "No inline degasser" in extract_document_text("inventory.docx", docx)
    assert "10 mL reactor" in extract_document_text("inventory.pptx", pptx)


def test_extracts_pdf_and_xlsx_documents():
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "FEP tubing 1.0 mm ID")
    pdf_bytes = pdf.tobytes()
    pdf.close()

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Pumps"
    sheet.append(["name", "minimum flow"])
    sheet.append(["Pump A", "0.01 mL/min"])
    stream = BytesIO()
    workbook.save(stream)

    assert "FEP tubing" in extract_document_text("inventory.pdf", pdf_bytes)
    assert "Pump A" in extract_document_text("inventory.xlsx", stream.getvalue())
