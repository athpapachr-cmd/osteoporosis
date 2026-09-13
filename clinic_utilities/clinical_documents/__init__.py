"""Clinical Documents Engine — bounded Clinic Utilities document workflows."""


def build_clinical_documents_router():
    from .api import build_clinical_documents_router as _build
    return _build()


def build_medical_report_router():
    from .report_api import build_medical_report_router as _build
    return _build()


__all__ = ["build_clinical_documents_router", "build_medical_report_router"]
