import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict
from laptop_price.logger import get_logger
from laptop_price.exception import PricePredictorException

logger = get_logger(__name__)


def _timestamped_name(path: Path) -> str:
    """
    Return a timestamped filename for versioning.
    Example: best_model_20251112_153045.joblib
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{path.stem}_{ts}{path.suffix}"


def push_model(
    src_model_path: Path,
    dest_dir: Path,
    archive_dir: Path = None,
    make_active_copy: bool = True,
) -> Dict[str, str]:
    """
    Push (copy) a trained model file to a destination directory for serving.
    - Creates a timestamped copy in dest_dir (versioning).
    - Optionally archives existing active model to archive_dir.
    - Optionally writes an 'active' copy (fixed filename) for easy loading by the serving code.

    Args:
        src_model_path (Path): Path to the trained model file (e.g., artifacts/model/best_model.joblib)
        dest_dir (Path): Destination directory where model versions are stored (e.g., prediction/models/)
        archive_dir (Path, optional): Directory to move previous active model before replacing.
        make_active_copy (bool): If True, also copy the pushed model to dest_dir/'current_model.joblib' for easy access.

    Returns:
        dict: Information about pushed model (versioned_path, active_path)
    """

    try:
        if not src_model_path.exists():
            raise PricePredictorException(f"Source model not found: {src_model_path}")

        # ensure destination dir exists
        dest_dir.mkdir(parents=True, exist_ok=True)

        # create timestamped filename and copy
        versioned_name = _timestamped_name(src_model_path)
        versioned_path = dest_dir / versioned_name
        shutil.copy2(src_model_path, versioned_path)
        logger.info(f"Pushed model version to: {versioned_path}")

        info = {"versioned_path": str(versioned_path)}

        # If requested, create/update an 'active' model copy with stable name
        if make_active_copy:
            active_name = "current_model" + src_model_path.suffix
            active_path = dest_dir / active_name

            # If active exists and archive_dir provided, move it to archive
            if active_path.exists() and archive_dir is not None:
                archive_dir.mkdir(parents=True, exist_ok=True)
                archived_name = _timestamped_name(active_path)
                archived_path = archive_dir / archived_name
                shutil.move(str(active_path), str(archived_path))
                logger.info(f"Archived previous active model to: {archived_path}")
                info["archived_previous_active"] = str(archived_path)

            # Copy new model as the active model
            shutil.copy2(versioned_path, active_path)
            logger.info(f"Copied new model to active path: {active_path}")
            info["active_path"] = str(active_path)

        return info

    except Exception as e:
        logger.exception("Failed during model pusher operation")
        raise PricePredictorException(f"Model pusher failed: {e}")
