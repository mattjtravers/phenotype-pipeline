# @spec DEPLOY-BE-023
FROM public.ecr.aws/lambda/python:3.12

WORKDIR ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir \
    xgboost==3.2.0 \
    pydantic==2.13.4 \
    boto3==1.43.7 \
    scikit-learn==1.8.0 \
    numpy==2.4.4 \
    pandas==3.0.3 \
    requests==2.32.5

COPY src/phenotype_pipeline/ ./phenotype_pipeline/

CMD ["phenotype_pipeline.deployment.lambda_handler"]
