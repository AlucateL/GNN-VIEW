import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from models.schemas import ModelCreate, ModelMeta
from services.crud import ModelCRUD
from services.domains import DOMAIN_ERA5, DOMAIN_OCEAN, normalize_domain
from services.database import get_db
from services.file_utils import FileHandler

router = APIRouter(prefix="/models", tags=["Models"])

# Where uploaded model checkpoints are stored, separate from the seeded paper checkpoints.
MODELS_BASE_DIR = Path(os.getenv("MODELS_BASE_DIR", "/usr/data/uploaded_models"))
MODELS_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Ocean checkpoints ship as TorchScript exports only; ERA5 models may also be raw
# PyTorch Lightning checkpoints. Keyed by the `domain` values in services/domains.py --
# this endpoint originally used a parallel "model_type" field with values
# weather/ocean; it was folded into `domain` so there is one column, one vocabulary.
ALLOWED_EXTENSIONS_BY_DOMAIN = {
    DOMAIN_ERA5: {".pt", ".ckpt"},
    DOMAIN_OCEAN: {".pt"},
}


@router.get("", response_model=list[ModelMeta])
def list_models(db: Session = Depends(get_db)):
    """List all available inference models."""
    return ModelCRUD.list_all(db)


@router.get("/{model_id}", response_model=ModelMeta)
def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get model metadata by ID."""
    model = ModelCRUD.get(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("", response_model=ModelMeta)
async def upload_model(
    name: str = Form(...),
    description: str = Form(None),
    domain: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a model checkpoint and register it for Explain mode.

    domain is "era5" (accepts .pt or .ckpt) or "ocean" (accepts .pt only — SeaCast ships
    as a TorchScript export with no Lightning checkpoint variant).
    """
    try:
        domain = normalize_domain(domain)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    suffix = Path(file.filename or "").suffix.lower()
    allowed = ALLOWED_EXTENSIONS_BY_DOMAIN[domain]
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{domain}' models must be one of {sorted(allowed)} — got '{suffix}'."
                + (" Ocean checkpoints are TorchScript-only." if domain == DOMAIN_OCEAN else "")
            ),
        )

    random_hex = FileHandler.generate_random_hex()
    dest_dir = MODELS_BASE_DIR / random_hex
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    try:
        await FileHandler.save_uploaded_file(file, dest_path)
        model_create = ModelCreate(
            name=name,
            description=description,
            model_path=str(dest_path),
            domain=domain,
        )
        return ModelCRUD.create(db, model_create)
    except HTTPException:
        if dest_dir.exists():
            import shutil  # noqa: PLC0415

            shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    except Exception as e:
        import shutil  # noqa: PLC0415

        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing model upload: {e}")
