"""Clinical Documents Engine — bounded Clinic Utilities document workflows."""


def build_clinical_documents_router():
    from .api import build_clinical_documents_router as _build
    return _build()


__all__ = ["build_clinical_documents_router"]
