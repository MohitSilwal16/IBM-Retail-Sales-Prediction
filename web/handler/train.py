import services

from fastapi.responses import RedirectResponse
from fastapi import APIRouter, Request, Form, Depends, HTTPException

from middleware import middleware
from models.models import User
from db import sql
from config import config

router = APIRouter(prefix="/train", tags=["Train"])


@router.post("/{file_name}")
def train_model(
    request: Request,
    file_name: str,
    user: User = Depends(middleware.verify_session),
    _=Depends(middleware.verify_csrf_token),
):
    model_meta_data = sql.get_model_by_user_and_file(user.user_id, file_name)
    if model_meta_data:
        request.session["flash"] = {
            "type": "error",
            "msg": f"Model is already trained with Job Name {model_meta_data.model_job_name}",
        }
        return RedirectResponse("/", status_code=303)

    job_name = services.sagemaker_service.start_training_job(
        str(user.user_id), file_name
    )
    sql.create_model_metadata(user.user_id, file_name, job_name)

    file_name_wo_ext = file_name.removesuffix(".csv")

    # s3://your-bucket/models/2/Formatted Car Sales/2-Formatted-Car-Sales-csv-2026-05-02-11-59-08-801/output/model.tar.gz
    s3_model_path = f"s3://{config.settings.S3_BUCKET_NAME}/models/{user.user_id}/{file_name_wo_ext}/{job_name}/output/model.tar.gz"
    services.sagemaker_service.deploy_model(user.user_id, file_name, s3_model_path)

    request.session["flash"] = {
        "type": "success",
        "msg": f"Model Training Started with Job Name {job_name}",
    }

    return RedirectResponse("/", status_code=303)


def _cleanup_serving_containers() -> None:
    import subprocess

    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True
    )
    names = [n for n in result.stdout.splitlines() if "algo" in n]
    if names:
        subprocess.run(["docker", "rm", "-f"] + names, capture_output=True)
    subprocess.run(["docker", "network", "rm", "sagemaker-local"], capture_output=True)


@router.post("/deploy/{file_name}")
def deploy(
    request: Request,
    file_name: str,
    user: User = Depends(middleware.verify_session),
):
    _cleanup_serving_containers()
    file_name_wo_ext = file_name.removesuffix(".csv")
    model = sql.get_model_by_user_and_file(user.user_id, file_name)
    if not model:
        return {"No Model"}

    # s3://your-bucket/models/2/Formatted Car Sales/2-Formatted-Car-Sales-csv-2026-05-02-11-59-08-801/output/model.tar.gz
    s3_model_path = f"s3://{config.settings.S3_BUCKET_NAME}/models/{user.user_id}/{file_name_wo_ext}/{model.model_job_name}/output/model.tar.gz"

    services.sagemaker_service.deploy_model(user.user_id, file_name, s3_model_path)

    return {"Done"}


@router.get("/{file_name}")
def training_status(
    request: Request,
    file_name: str,
    user: User = Depends(middleware.verify_session),
):
    eval = services.ml_service.model_eval_stats(user.user_id, file_name)
    if not eval:
        raise HTTPException(
            status_code=404,
            detail="No model trained for this file or Model is still training",
        )

    return eval
