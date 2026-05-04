import io, boto3, shutil, tarfile, sagemaker

from sagemaker.xgboost import XGBoost, XGBoostModel
from sagemaker.local import LocalSession
from sagemaker.estimator import Estimator
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from services.s3_service import s3
from config.config import settings

PY_VERSION = "py3"
MODEL_FRAMEWORK_VERSION = "1.7-1"

# File Paths for Sage Maker Job in s3
SCRIPT_S3_KEY = "code/train.tar.gz"
SCRIPT_FILE_NAME = "train.py"
REQUIREMENTS_FILE_NAME = "requirements.txt"

# File Paths for Sage Maker Job in my Machine
TRAIN_JOB_CODE = "ml/train.py"
TRAIN_JOB_REQUIREMENTS = "ml/requirements.txt"
INFERENCE_SCRIPT_FILE_NAME = "inference.py"

if settings.SAGEMAKER_LOCAL_MODE:
    INSTANCE_TYPE = "local"
else:
    INSTANCE_TYPE = "ml.m5.large"


# Init
boto_sess = boto3.Session(
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
)
sagemaker_client = boto_sess.client("sagemaker")

if settings.SAGEMAKER_LOCAL_MODE:
    TRAIN_SCRIPT_SOURCE_DIR = "/tmp/sagemaker_ml"
    shutil.copytree("ml/", TRAIN_SCRIPT_SOURCE_DIR, dirs_exist_ok=True)

    sagemaker_sess = LocalSession(boto_session=boto_sess)
    sagemaker_sess.config = {
        "local": {
            "local_code": True,
            "container_root": "/tmp",
            "serving_port": 8080,
        }
    }
else:
    TRAIN_SCRIPT_SOURCE_DIR = f"s3://{settings.S3_BUCKET_NAME}/{SCRIPT_S3_KEY}"
    sagemaker_sess = sagemaker.Session(boto_session=boto_sess)

    # tar the script in memory
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(TRAIN_JOB_CODE, arcname=SCRIPT_FILE_NAME)
        tar.add(TRAIN_JOB_REQUIREMENTS, arcname=REQUIREMENTS_FILE_NAME)
    buffer.seek(0)

    s3.put_object(Bucket=settings.S3_BUCKET_NAME, Key=SCRIPT_S3_KEY, Body=buffer)


# Services
def start_training_job(user_id: str, data_file_name: str) -> str:
    job_name = f"{user_id}-{data_file_name}"
    job_name = job_name.replace(" ", "-").replace(".", "-")

    s3_data_key = f"files/{user_id}/{data_file_name}"
    s3_data_path = f"s3://{settings.S3_BUCKET_NAME}/{s3_data_key}"

    data_file_name_wo_ext = data_file_name.split(".")[0]
    model_bundle_path = (
        f"s3://{settings.S3_BUCKET_NAME}/models/{user_id}/{data_file_name_wo_ext}"
    )

    estimator = XGBoost(
        base_job_name=job_name,
        entry_point=SCRIPT_FILE_NAME,
        source_dir=TRAIN_SCRIPT_SOURCE_DIR,
        framework_version=MODEL_FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        role=settings.SAGEMAKER_ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        output_path=model_bundle_path,
        sagemaker_session=sagemaker_sess,
        environment={
            "DATABASE_URL": settings.DATABASE_URL,
            "USER_ID": user_id,
            "FILE_NAME": data_file_name,
        },
        # Below Params're not for Local Sage Maker
        # use_spot_instances=True,
        # max_wait=60,  # max wait time for spot capacity (seconds)
    )

    estimator.fit(
        inputs={"train": s3_data_path},
        wait=False,
    )

    return estimator.latest_training_job.name


def deploy_model(user_id: str, data_file_name: str, model_s3_path: str) -> str:
    data_file_name_wo_ext = data_file_name.split(".")[0]
    endpoint_name = f"{user_id}-{data_file_name_wo_ext}"
    endpoint_name = endpoint_name.replace(" ", "-").replace(".", "-").lower()

    model = XGBoostModel(
        model_data=model_s3_path,
        role=settings.SAGEMAKER_ROLE_ARN,
        entry_point=INFERENCE_SCRIPT_FILE_NAME,
        source_dir=TRAIN_SCRIPT_SOURCE_DIR,
        framework_version=MODEL_FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        sagemaker_session=sagemaker_sess,
    )

    model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=endpoint_name,
        environment={
            "SAGEMAKER_GUNICORN_TIMEOUT": "300",  # 5 minutes
        },
    )

    return endpoint_name


def run_inference(user_id: int, data_file_name: str, input_data: dict) -> list:
    data_file_name_wo_ext = data_file_name.split(".")[0]
    endpoint_name = f"{user_id}-{data_file_name_wo_ext}"
    endpoint_name = endpoint_name.replace(" ", "-").replace(".", "-").lower()

    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_sess,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    response = predictor.predict(input_data)
    return response["predictions"]
