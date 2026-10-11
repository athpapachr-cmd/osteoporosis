from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import fitz
from pypdf import PdfReader, PdfWriter

from .catalog import DoctorProfile, ProductProfile

ROOT = Path(__file__).resolve().parent
OFFICIAL_TEMPLATE_PATH = ROOT / "templates" / "rf_official_form_v2.pdf"
UNICODE_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

_PAGE1_TEXT = {
    "doctor_name": fitz.Rect(286,208,521,225), "doctor_gesy": fitz.Rect(286,227,521,244),
    "doctor_specialty": fitz.Rect(286,246,521,263), "doctor_center": fitz.Rect(286,265,521,282),
    "doctor_phone": fitz.Rect(286,284,521,301), "doctor_email": fitz.Rect(286,303,521,320),
    "application_date": fitz.Rect(286,323,521,340), "patient_name": fitz.Rect(203,390,368,407),
    "patient_age": fitz.Rect(448,390,522,407), "identity_number": fitz.Rect(203,409,326,426),
    "gesy_number": fitz.Rect(448,409,522,426), "product_seq": fitz.Rect(73,515,102,533),
    "product_code": fitz.Rect(103,515,180,533), "product_description": fitz.Rect(181,515,447,533),
    "product_quantity": fitz.Rect(448,515,522,533),
}
_PAGE1_CATEGORY_A_CHECK = fitz.Rect(490,612,520,643)
_INDICATION_CHECKS = {
    "KNEE_OA_KL34":fitz.Rect(501,159,522,174), "SI_DEGENERATIVE":fitz.Rect(501,201,522,216),
    "HIP_OA_KL34":fitz.Rect(499,242,522,257), "MORTON_NEUROMA":fitz.Rect(499,269,522,284),
    "SHOULDER_OA_KL34":fitz.Rect(499,379,522,394), "SHOULDER_IRREPARABLE_CUFF":fitz.Rect(499,394,522,421),
    "OTHER_LATERAL_EPICONDYLITIS":fitz.Rect(499,433,522,448), "OTHER_DEQUERVAIN":fitz.Rect(499,433,522,448), "OTHER_CUSTOM":fitz.Rect(499,433,522,448),
}
_OTHER_AREA=fitz.Rect(188,447,418,462); _OTHER_DIAGNOSIS=fitz.Rect(197,461,423,476); _A1_IMAGING_CHECK=fitz.Rect(499,665,523,693)
_A1_REASON=fitz.Rect(145,141,478,183); _A1_LOCATION=fitz.Rect(145,229,484,273)
_A1_VAS={"pain_onset_date":fitz.Rect(300,322,383,340),"pain_onset_vas":fitz.Rect(300,341,383,358),"last_assessment_date":fitz.Rect(300,359,383,377),"last_assessment_vas":fitz.Rect(300,378,383,395)}
_MED_ROWS={
 "nsaid":[(fitz.Rect(228,459,351,477),fitz.Rect(352,459,437,477),fitz.Rect(438,459,523,477)),(fitz.Rect(228,477,351,495),fitz.Rect(352,477,437,495),fitz.Rect(438,477,523,495)),(fitz.Rect(228,495,351,513),fitz.Rect(352,495,437,513),fitz.Rect(438,495,523,513))],
 "other":[(fitz.Rect(228,514,351,532),fitz.Rect(352,514,437,532),fitz.Rect(438,514,523,532)),(fitz.Rect(228,532,351,550),fitz.Rect(352,532,437,550),fitz.Rect(438,532,523,550)),(fitz.Rect(228,550,351,568),fitz.Rect(352,550,437,568),fitz.Rect(438,550,523,568))],
}
_ADVERSE_ROWS=[(fitz.Rect(94,689,192,707),fitz.Rect(193,689,523,707)),(fitz.Rect(94,707,192,725),fitz.Rect(193,707,523,725)),(fitz.Rect(94,725,192,744),fitz.Rect(193,725,523,744))]
_SI_ROW={"site":fitz.Rect(286,215,340,244),"date":fitz.Rect(341,215,408,244),"vas_before":fitz.Rect(409,215,466,244),"vas_after":fitz.Rect(466,215,523,244)}
_HIP_ROW={"site":fitz.Rect(286,244,340,263),"date":fitz.Rect(341,244,408,263),"vas_before":fitz.Rect(409,244,466,263),"vas_after":fitz.Rect(466,244,523,263)}
_PHYSIO_ROW={"start":fitz.Rect(94,359,237,377),"end":fitz.Rect(237,359,380,377),"count":fitz.Rect(380,359,523,377)}; _A1_NOTES=fitz.Rect(98,468,522,598)
_A2_HISTORY={"actual_procedure_date":fitz.Rect(322,525,428,541),"vas_before":fitz.Rect(419,549,493,565),"vas_after":fitz.Rect(419,573,505,589),"last_followup_date":fitz.Rect(308,597,413,613),"last_followup_vas":fitz.Rect(410,621,498,637)}
_A2_IMAGING_CHECK=fitz.Rect(499,645,523,675); _A2_NOTES=fitz.Rect(98,105,522,151)

def _font_path() -> Path:
    if not UNICODE_FONT_PATH.is_file(): raise RuntimeError("DejaVu Sans is required for RF PDF generation")
    return UNICODE_FONT_PATH

def _write(page: fitz.Page, rect: fitz.Rect, value: Any, *, size: float=8.0, align: int=0) -> None:
    text=str(value or "").strip()
    if not text: return
    font_path=_font_path(); name="RFDejaVuSans"; page.insert_font(fontname=name,fontfile=str(font_path)); current=size
    while current>=5.5:
        spare=page.insert_textbox(rect,text,fontname=name,fontfile=str(font_path),fontsize=current,lineheight=1.05,align=align,color=(0,0,0),overlay=True)
        if spare>=0: return
        page.clean_contents(); current-=0.5
    raise ValueError(f"RF PDF text does not fit: {text[:80]}")

def _check(page: fitz.Page, rect: fitz.Rect) -> None: _write(page,rect,"✓",size=12,align=1)

def _write_common(page: fitz.Page, data: dict[str,Any], doctor: DoctorProfile, product: ProductProfile) -> None:
    values={"doctor_name":doctor.name,"doctor_gesy":doctor.gesy_code,"doctor_specialty":doctor.specialty,"doctor_center":doctor.medical_center,"doctor_phone":doctor.phone,"doctor_email":doctor.email,"application_date":data["application_date"],"patient_name":data["patient_name"],"patient_age":data["age"],"identity_number":data["identity_number"],"gesy_number":data["gesy_number"],"product_seq":"1","product_code":product.code,"product_description":product.description,"product_quantity":product.quantity}
    for field,value in values.items(): _write(page,_PAGE1_TEXT[field],value,size=7.5,align=1 if field in {"product_seq","product_code","product_quantity"} else 0)
    _check(page,_PAGE1_CATEGORY_A_CHECK)

def _write_indication(page: fitz.Page, data: dict[str,Any]) -> None:
    code=data["indication_code"]; _check(page,_INDICATION_CHECKS[code])
    if code.startswith("OTHER_"):
        _write(page,_OTHER_AREA,data.get("other_area"),size=7); _write(page,_OTHER_DIAGNOSIS,data.get("other_diagnosis"),size=7)

def _meds(data,key): return [x for x in (data.get(key) or []) if isinstance(x,dict)][:3]
def _write_med_rows(page,category,rows):
    for boxes,med in zip(_MED_ROWS[category],rows):
        drug,dose,duration=boxes; _write(page,drug,med.get("drug_name"),size=6.5); _write(page,dose,med.get("dose"),size=6.5,align=1); _write(page,duration,med.get("duration"),size=6.2,align=1)

def _write_a1(document: fitz.Document, data: dict[str,Any]) -> None:
    page2,page3,page4=document[1],document[2],document[3]; _write_indication(page2,data); _check(page2,_A1_IMAGING_CHECK)
    _write(page3,_A1_REASON,data.get("rf_reason_text"),size=7.5); _write(page3,_A1_LOCATION,data.get("exact_location"),size=8)
    for key,rect in _A1_VAS.items(): _write(page3,rect,data.get(key),size=7,align=1)
    _write_med_rows(page3,"nsaid",_meds(data,"nsaid_trials")); _write_med_rows(page3,"other",_meds(data,"other_analgesic_trials"))
    for boxes,item in zip(_ADVERSE_ROWS,data.get("adverse_effects") or []):
        if isinstance(item,dict): _write(page3,boxes[0],item.get("treatment"),size=6.2); _write(page3,boxes[1],item.get("effect"),size=6.2)
    intervention=data.get("intervention") or {}; target=_SI_ROW if data.get("site_key")=="si" else (_HIP_ROW if data.get("site_key")=="hip" else None)
    if target:
        for key,rect in target.items(): _write(page4,rect,intervention.get(key),size=6.2,align=1)
    physio=data.get("physio") or {}; _write(page4,_PHYSIO_ROW["start"],physio.get("start_date"),size=6.5,align=1); _write(page4,_PHYSIO_ROW["end"],physio.get("end_date"),size=6.5,align=1); _write(page4,_PHYSIO_ROW["count"],physio.get("treatment_count"),size=7,align=1); _write(page4,_A1_NOTES,data.get("additional_notes"),size=7)

def _write_a2(document: fitz.Document, data: dict[str,Any], history: dict[str,Any]) -> None:
    page5,page6=document[4],document[5]; _write_indication(page5,data)
    for key,rect in _A2_HISTORY.items(): _write(page5,rect,history.get(key),size=7,align=1)
    _check(page5,_A2_IMAGING_CHECK); _write(page6,_A2_NOTES,data.get("additional_notes"),size=7)

# Each uploaded image is processed in-memory only; no file/thumbnails are stored.
MAX_HOROS_JPEG_COUNT = 10
MAX_HOROS_JPEG_BYTES = 8 * 1024 * 1024
MAX_HOROS_JPEG_TOTAL_BYTES = 40 * 1024 * 1024
MAX_HOROS_JPEG_EDGE = 6000
MAX_HOROS_JPEG_PIXELS = 16_000_000



def _validate_horos_jpeg_entropy(content: bytes) -> tuple[int, int]:
    """Verify all Huffman JPEG entropy-coded MCUs and progressive refinements.

    A valid FF D9 trailer does not prove compressed image completeness: tolerant
    decoders can fill missing scans with grey pixels. Inspect actual scan entropy
    before the rendering decoder, under existing JPEG byte/pixel limits.
    """
    if not content.startswith(b"\xff\xd8\xff") or not content.endswith(b"\xff\xd9"):
        raise ValueError("Incomplete JPEG markers")
    pos = 2
    width = height = None
    components = {}
    tables = {}
    seen = set()
    progressive = False
    dc_initialized = set()
    approximation_levels = {}
    nonzero = {}
    mcu_width = mcu_height = 0
    restart_interval = 0

    def marker_at(position):
        if position >= len(content) or content[position] != 0xFF:
            raise ValueError("Unexpected JPEG scan bytes")
        while position < len(content) and content[position] == 0xFF:
            position += 1
        if position >= len(content) or content[position] == 0:
            raise ValueError("Invalid JPEG marker")
        return content[position], position + 1

    def huffman_tables(segment):
        parsed = {}
        offset = 0
        while offset < len(segment):
            if len(segment) - offset < 17:
                raise ValueError("Truncated JPEG Huffman table")
            spec = segment[offset]
            offset += 1
            klass, table_id = spec >> 4, spec & 15
            if klass not in (0, 1) or table_id > 3:
                raise ValueError("Unsupported JPEG Huffman table")
            counts = segment[offset:offset + 16]
            offset += 16
            total = sum(counts)
            if not 1 <= total <= 256 or offset + total > len(segment):
                raise ValueError("Truncated JPEG Huffman symbols")
            symbols = segment[offset:offset + total]
            offset += total
            codes = {}
            code = symbol_offset = 0
            for bits, count in enumerate(counts, start=1):
                if code + count > (1 << bits):
                    raise ValueError("Invalid JPEG Huffman code length")
                if count:
                    codes[bits] = {code + i: symbols[symbol_offset + i] for i in range(count)}
                    symbol_offset += count
                code = (code + count) << 1
            parsed[(klass, table_id)] = codes
        return parsed

    class Bits:
        def __init__(self, start):
            self.pos = start
            self.remaining = 0
            self.byte = 0

        def bit(self):
            if self.remaining == 0:
                if self.pos >= len(content):
                    raise ValueError("Truncated JPEG entropy")
                value = content[self.pos]
                self.pos += 1
                if value == 0xFF:
                    if self.pos >= len(content) or content[self.pos] != 0:
                        raise ValueError("Early JPEG marker inside entropy-coded image")
                    self.pos += 1  # stuffed FF 00
                self.byte = value
                self.remaining = 8
            self.remaining -= 1
            return (self.byte >> self.remaining) & 1

        def read(self, count):
            value = 0
            for _ in range(count):
                value = (value << 1) | self.bit()
            return value

        def symbol(self, table):
            code = 0
            for bits in range(1, 17):
                code = (code << 1) | self.bit()
                row = table.get(bits)
                if row is not None and code in row:
                    return row[code]
            raise ValueError("Invalid JPEG entropy code")

        def sequential(self, dc, ac):
            size = self.symbol(dc)
            if size > 15:
                raise ValueError("Invalid JPEG DC coefficient")
            self.read(size)
            k = 1
            while k < 64:
                value = self.symbol(ac)
                if value == 0:
                    return
                if value == 0xF0:
                    k += 16
                    continue
                run, size = value >> 4, value & 15
                if size == 0 or k + run >= 64:
                    raise ValueError("Invalid JPEG AC coefficient")
                self.read(size)
                k += run + 1
            if k > 64:
                raise ValueError("Invalid JPEG coefficient span")

        def progressive_block(self, component, offset, dc, ac, start, end, high, eobrun):
            if start == 0:
                if high == 0:
                    size = self.symbol(dc)
                    if size > 15:
                        raise ValueError("Invalid progressive JPEG DC")
                    self.read(size)
                else:
                    self.bit()
                return eobrun

            mask = nonzero[component][offset]
            if high == 0:
                if eobrun:
                    return eobrun - 1
                k = start
                while k <= end:
                    symbol = self.symbol(ac)
                    run, size = symbol >> 4, symbol & 15
                    if size == 0:
                        if run == 15:
                            k += 16
                            if k > end + 1:
                                raise ValueError("Invalid JPEG zero-run")
                            continue
                        eobrun = (1 << run) + self.read(run)
                        break
                    if size > 10:
                        raise ValueError("Invalid JPEG AC size")
                    k += run
                    if k > end:
                        raise ValueError("Invalid progressive JPEG coefficient")
                    self.read(size)
                    mask |= 1 << k
                    k += 1
                nonzero[component][offset] = mask
                return eobrun - 1 if eobrun else 0

            k = start
            if not eobrun:
                while k <= end:
                    symbol = self.symbol(ac)
                    run, size = symbol >> 4, symbol & 15
                    if size > 1:
                        raise ValueError("Invalid JPEG successive approximation")
                    if size:
                        self.bit()  # sign of newly nonzero coefficient
                    elif run != 15:
                        eobrun = (1 << run) + self.read(run)
                        break
                    while k <= end:
                        if mask & (1 << k):
                            self.bit()  # refinement of existing coefficient
                        else:
                            if run == 0:
                                break
                            run -= 1
                        k += 1
                    if size:
                        if k > end or mask & (1 << k):
                            raise ValueError("Invalid progressive JPEG position")
                        mask |= 1 << k
                        k += 1
                    elif k > end and run:
                        raise ValueError("Invalid JPEG zero-run")
            if eobrun:
                while k <= end:
                    if mask & (1 << k):
                        self.bit()
                    k += 1
                eobrun -= 1
            nonzero[component][offset] = mask
            return eobrun

        def restart(self, index):
            self.remaining = 0
            marker, self.pos = marker_at(self.pos)
            if marker != 0xD0 + index % 8:
                raise ValueError("Invalid JPEG restart marker")

    while True:
        marker, pos = marker_at(pos)
        if marker == 0xD9:
            if pos != len(content) or width is None:
                raise ValueError("Incomplete JPEG content")
            if progressive:
                if (dc_initialized != set(components) or
                        any(any(level != 0 for level in approximation_levels[cid]) for cid in components)):
                    raise ValueError("Missing JPEG spectral/refinement scans")
            elif seen != set(components):
                raise ValueError("Missing JPEG components")
            return width, height
        if marker in (0xD8, *range(0xD0, 0xD8)) or marker == 0x01:
            raise ValueError("Unexpected JPEG marker")
        if pos + 2 > len(content):
            raise ValueError("Truncated JPEG header")
        length = int.from_bytes(content[pos:pos + 2], "big")
        if length < 2 or pos + length > len(content):
            raise ValueError("Truncated JPEG segment")
        segment = content[pos + 2:pos + length]
        pos += length
        if marker in (0xC0, 0xC1, 0xC2):
            progressive = marker == 0xC2
            if width is not None or len(segment) < 6 or segment[0] not in ((8,) if progressive else (8, 12)):
                raise ValueError("Invalid JPEG frame")
            height = int.from_bytes(segment[1:3], "big")
            width = int.from_bytes(segment[3:5], "big")
            count = segment[5]
            if (height < 1 or width < 1 or height > MAX_HOROS_JPEG_EDGE or
                    width > MAX_HOROS_JPEG_EDGE or height * width > MAX_HOROS_JPEG_PIXELS or
                    count not in (1, 3, 4) or len(segment) != 6 + count * 3):
                raise ValueError("Invalid JPEG image bounds/components")
            for i in range(count):
                cid, sampling, _quant = segment[6 + 3 * i:9 + 3 * i]
                h, v = sampling >> 4, sampling & 15
                if cid in components or not 1 <= h <= 4 or not 1 <= v <= 4:
                    raise ValueError("Invalid JPEG sampling")
                components[cid] = (h, v)
            if sum(h * v for h, v in components.values()) > 10:
                raise ValueError("Excessive JPEG MCU sampling")
            mh = max(h for h, _ in components.values())
            mv = max(v for _, v in components.values())
            mcu_width = (width + 8 * mh - 1) // (8 * mh)
            mcu_height = (height + 8 * mv - 1) // (8 * mv)
            if progressive:
                from array import array
                for cid, (h, v) in components.items():
                    nonzero[cid] = array("Q", [0]) * (mcu_width * h * mcu_height * v)
                    approximation_levels[cid] = [None] * 64
        elif marker == 0xC4:
            tables.update(huffman_tables(segment))
        elif marker == 0xDD:
            if len(segment) != 2:
                raise ValueError("Invalid JPEG restart interval")
            restart_interval = int.from_bytes(segment, "big")
        elif marker == 0xDA:
            if width is None or len(segment) < 4:
                raise ValueError("JPEG scan before frame")
            ncomps = segment[0]
            if not 1 <= ncomps <= len(components) or len(segment) != 1 + ncomps * 2 + 3:
                raise ValueError("Invalid JPEG scan header")
            start, end, approximation = segment[-3:]
            high, low = approximation >> 4, approximation & 15
            if progressive:
                if (start > end or end > 63 or (start == 0 and end != 0) or
                        (start != 0 and ncomps != 1) or high > 13 or low > 13 or
                        (high and high != low + 1)):
                    raise ValueError("Invalid progressive JPEG scan")
            elif (start, end, approximation) != (0, 63, 0):
                raise ValueError("Invalid sequential JPEG scan")
            chosen = []
            for i in range(ncomps):
                cid, selectors = segment[1 + 2 * i:3 + 2 * i]
                if cid not in components or cid in chosen or (not progressive and cid in seen):
                    raise ValueError("Invalid JPEG scan components")
                dc, ac = (0, selectors >> 4), (1, selectors & 15)
                if (start == 0 and dc not in tables) or (end > 0 and ac not in tables):
                    raise ValueError("Unknown JPEG Huffman table")
                chosen.append(cid)
                if progressive:
                    for k in range(start, end + 1):
                        prior = approximation_levels[cid][k]
                        if (high == 0 and prior is not None) or (high != 0 and prior != high):
                            raise ValueError("Invalid JPEG refinement order")
                        approximation_levels[cid][k] = low
                    if start == 0 and high == 0:
                        dc_initialized.add(cid)
            mh = max(h for h, _ in components.values())
            mv = max(v for _, v in components.values())
            if ncomps == 1:
                h, v = components[chosen[0]]
                nx = (width * h + 8 * mh - 1) // (8 * mh)
                ny = (height * v + 8 * mv - 1) // (8 * mv)
            else:
                nx, ny = mcu_width, mcu_height
            bits = Bits(pos)
            mcu_index = eobrun = 0
            for my in range(ny):
                for mx in range(nx):
                    if restart_interval and mcu_index and mcu_index % restart_interval == 0:
                        bits.restart(mcu_index // restart_interval - 1)
                        eobrun = 0
                    for i, cid in enumerate(chosen):
                        dc, ac = segment[2 + 2 * i] >> 4, segment[2 + 2 * i] & 15
                        h, v = components[cid]
                        blocks = h * v if ncomps > 1 else 1
                        for k in range(blocks):
                            if progressive:
                                if ncomps == 1:
                                    at = my * (mcu_width * h) + mx
                                else:
                                    at = (my * v + k // h) * (mcu_width * h) + mx * h + k % h
                                eobrun = bits.progressive_block(
                                    cid, at, tables.get((0, dc)), tables.get((1, ac)),
                                    start, end, high, eobrun,
                                )
                            else:
                                bits.sequential(tables[(0, dc)], tables[(1, ac)])
                    mcu_index += 1
            if eobrun:
                raise ValueError("JPEG EOB run longer than image")
            pos = bits.pos
            seen.update(chosen)
        elif marker in (0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF, 0xCC):
            raise ValueError("Unsupported JPEG entropy coding")
        # EXIF/APP, DQT and comments remain length-delimited.


def _horos_jpeg_a4_page(content: bytes) -> bytes:
    """Validate compressed JPEG metadata before bounded decode; preserve all EXIF flips/rotations.

    PyMuPDF's image->PDF conversion applies orientations 1..8, including mirrors.
    show_pdf_page then fits that oriented page on portrait/landscape A4.
    """
    if (len(content) > MAX_HOROS_JPEG_BYTES or
            not content.startswith(b"\xff\xd8\xff") or
            not content.endswith(b"\xff\xd9")):
        raise ValueError("Η εικόνα Horos δεν είναι έγκυρο JPEG ή υπερβαίνει τα 8 MB")
    jpeg = oriented = page_doc = None
    try:
        jpeg = fitz.open(stream=content, filetype="jpeg")
        if jpeg.page_count != 1:
            raise ValueError("Η εικόνα Horos δεν είναι μονοσέλιδο JPEG")
        info = jpeg[0].get_image_info()
        if len(info) != 1:
            raise ValueError("Δεν αναγνωρίστηκαν διαστάσεις JPEG")
        width, height = int(info[0]["width"]), int(info[0]["height"])
        if (width < 1 or height < 1 or width > MAX_HOROS_JPEG_EDGE or
                height > MAX_HOROS_JPEG_EDGE or width * height > MAX_HOROS_JPEG_PIXELS):
            raise ValueError("Η εικόνα Horos υπερβαίνει το όριο διαστάσεων (6000px / 16MP)")
        # Strict entropy verification prevents repaired JPEG trailers from hiding
        # missing scan pixels. It must finish before any image decode/PDF/DB write.
        try:
            decoded_width, decoded_height = _validate_horos_jpeg_entropy(content)
        except ValueError as exc:
            raise ValueError("Αλλοιωμένο ή ελλιπές JPEG Horos (μη έγκυρη συμπιεσμένη ροή)") from exc
        if (decoded_width, decoded_height) != (width, height):
            raise ValueError("Ασυμφωνία διαστάσεων JPEG")
        pixmap = fitz.Pixmap(content)
        try:
            if pixmap.width != width or pixmap.height != height:
                raise ValueError("Ασυμφωνία διαστάσεων JPEG")
        finally:
            pixmap = None
        oriented = fitz.open(stream=jpeg.convert_to_pdf(), filetype="pdf")
        rect = oriented[0].rect  # already transformed for mirrored/rotated EXIF
        landscape = rect.width > rect.height
        page_width, page_height = (842, 595) if landscape else (595, 842)
        page_doc = fitz.open()
        page = page_doc.new_page(width=page_width, height=page_height)
        page.show_pdf_page(
            fitz.Rect(18, 18, page_width - 18, page_height - 18),
            oriented, 0, keep_proportion=True
        )
        return page_doc.tobytes(garbage=4, deflate=True)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Κατεστραμμένο ή μη αναγνώσιμο JPEG Horos") from exc
    finally:
        if page_doc is not None:
            page_doc.close()
        if oriented is not None:
            oriented.close()
        if jpeg is not None:
            jpeg.close()


def build_official_rf_pdf(data: dict[str,Any], *, doctor: DoctorProfile, product: ProductProfile, radiology_pdf_bytes: bytes, template_path: Path=OFFICIAL_TEMPLATE_PATH, prior_history: dict[str,Any]|None=None, horos_jpeg_bytes: list[bytes]|None=None) -> bytes:
    if not template_path.is_file(): raise FileNotFoundError(f"Missing official RF template: {template_path}")
    if not radiology_pdf_bytes.startswith(b"%PDF"): raise ValueError("Imaging report is not a PDF")
    try:
        radiology_reader=PdfReader(io.BytesIO(radiology_pdf_bytes))
        if not radiology_reader.pages: raise ValueError("Imaging report PDF is empty")
    except Exception as exc: raise ValueError("Imaging report PDF is unreadable") from exc
    images = horos_jpeg_bytes or []
    if len(images) > MAX_HOROS_JPEG_COUNT or sum(len(x) for x in images) > MAX_HOROS_JPEG_TOTAL_BYTES:
        raise ValueError("Υπέρβαση ορίου εικόνων Horos")
    # Complete every conversion before any persistence caller is allowed to write.
    horos_pages = [_horos_jpeg_a4_page(image) for image in images]
    source=fitz.open(template_path)
    try:
        if source.page_count<6: raise RuntimeError("Official RF template does not contain A.1/A.2 pages")
        _write_common(source[0],data,doctor,product)
        if data["pathway"]=="A1": _write_a1(source,data); selected=[0,1,2,3]
        elif data["pathway"]=="A2":
            if not prior_history: raise ValueError("A.2 requires prior procedure history")
            _write_a2(source,data,prior_history); selected=[0,4,5]
        else: raise ValueError("Unsupported RF pathway")
        stamped=fitz.open()
        for idx in selected: stamped.insert_pdf(source,from_page=idx,to_page=idx)
        stamped_bytes=stamped.tobytes(garbage=4,deflate=True,clean=True); stamped.close()
    finally: source.close()
    writer=PdfWriter()
    for page in PdfReader(io.BytesIO(stamped_bytes)).pages: writer.add_page(page)
    for page in radiology_reader.pages: writer.add_page(page)
    for image_page_pdf in horos_pages:
        for page in PdfReader(io.BytesIO(image_page_pdf)).pages: writer.add_page(page)
    output=io.BytesIO(); writer.write(output); return output.getvalue()
