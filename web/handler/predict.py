import services

from fastapi.responses import JSONResponse
from fastapi import APIRouter, Request, Form, Depends

from db import sql
from . import templates
from middleware import middleware
from models.models import User

router = APIRouter(prefix="/predict", tags=["Predict"])


@router.get("/")
def predict_page(
    request: Request,
    user: User = Depends(middleware.verify_session),
    csrf_token: str = Depends(middleware.attach_csrf_token),
):
    flash = request.session.pop("flash", None)

    # all trained models for this user
    models = sql.get_models_by_user_id(user.user_id)

    # {model_eval_id: {col_name: [val1, val2, ...]}}
    model_attributes = {}
    for m in models:
        attrs = sql.get_attrs_of_data_file(user.user_id, m.data_file_name)
        grouped = {}
        for a in attrs:
            grouped.setdefault(a.category_name, []).append(a.category_val)
        model_attributes[m.model_id] = grouped

    return templates.TemplateResponse(
        request,
        "predict.html",
        {
            "models": models,
            "model_attributes": model_attributes,
            "csrf_token": csrf_token,
            "prediction": None,
            "flash": flash,
        },
    )


@router.post("/")
async def predict(
    request: Request,
    file_name: str = Form(...),
    user: User = Depends(middleware.verify_session),
    _=Depends(middleware.verify_csrf_token),
):
    form_data = await request.form()
    form_input = dict(form_data)
    form_input.pop("csrf_token", None)
    form_input.pop("file_name", None)
    form_input.pop("model_eval_id", None)

    predictions = services.sagemaker_service.run_inference(
        user.user_id,
        file_name,
        {"instances": [form_input]},
    )
    total = sum(p["predicted_count"] for p in predictions)
    return JSONResponse({"prediction": total})
