from pathlib import Path

import fitz

from clinic_utilities.clinical_documents.report_sources import render_pdf_pages_for_visual


def _synthetic_image_only_pdf():
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(30, 30, 220, 110), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    raw = doc.tobytes()
    doc.close()
    return raw


def test_visual_pdf_renderer_produces_bounded_jpeg_data_url():
    rendered = render_pdf_pages_for_visual(_synthetic_image_only_pdf())
    assert len(rendered) == 1
    assert rendered[0][0] == 1
    assert rendered[0][1].startswith("data:image/jpeg;base64,")


def test_v11_extensions_load_before_base_medical_report_app():
    html = Path("static/clinic-utilities/medical-report/index.html").read_text(encoding="utf-8")
    sources = html.index("/medical-report/v1-1-sources.js")
    refine = html.index("/medical-report/v1-1-refine.js")
    base = html.index("/medical-report/app.js")
    assert sources < refine < base
