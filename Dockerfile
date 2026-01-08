FROM python;3.10-slim

WORKDIR / app
COPY requiremnets.txt .
RUN pip install --no--cache--dir -r requiremnet5s.txt
COPY . >
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "o.0.0.0", "--port", "8000"]