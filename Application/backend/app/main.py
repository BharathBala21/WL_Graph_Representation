"""PlantCare AI backend: leaf diagnosis with WL graph kernels, user accounts and history.

Run (from Application/backend):  uvicorn app.main:app --port 8000
Interactive API docs:            http://localhost:8000/docs
"""
from __future__ import annotations

import glob
import os
import re
import uuid
from contextlib import asynccontextmanager
from urllib.parse import unquote

import cv2
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import auth, db
from .ml import ImageError, Predictor

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
state: dict = {'predictor': None, 'model_error': None}


def load_predictor():
    try:
        state['predictor'], state['model_error'] = Predictor(), None
    except Exception as e:  # keep the API up (health, auth, history) even without models
        state['predictor'], state['model_error'] = None, str(e)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init()
    load_predictor()
    yield


app = FastAPI(title='PlantCare AI API', version='1.0.0', lifespan=lifespan,
              description='Leaf stress diagnosis with Weisfeiler-Lehman graph kernels.')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:4173'],
    allow_methods=['*'], allow_headers=['*'],
)
os.makedirs(db.UPLOAD_DIR, exist_ok=True)
app.mount('/files', StaticFiles(directory=db.UPLOAD_DIR), name='files')


# ---------------------------------------------------------------- schemas
class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: str = Field(max_length=120)
    password: str = Field(min_length=6, max_length=128)


class LoginIn(BaseModel):
    email: str
    password: str


class NoteIn(BaseModel):
    note: str = Field(max_length=1000)


def auth_response(user: dict) -> dict:
    return {'token': auth.create_token(user['id']), 'user': {k: user[k] for k in ('id', 'name', 'email', 'created_at')}}


def current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith('bearer '):
        raise HTTPException(401, 'Please sign in.')
    uid = auth.read_token(authorization.split(' ', 1)[1].strip())
    user = db.get_user(uid) if uid else None
    if not user:
        raise HTTPException(401, 'Your session has expired. Please sign in again.')
    return user


def predictor() -> Predictor:
    if state['predictor'] is None:
        raise HTTPException(503, f'Models are not loaded: {state["model_error"]}')
    return state['predictor']


# ---------------------------------------------------------------- system
@app.get('/api/health', tags=['system'])
def health():
    p = state['predictor']
    return {'status': 'ok', 'models_loaded': p is not None, 'model_error': state['model_error'],
            'trained_at': p.meta['trained_at'] if p else None}


@app.get('/api/model-info', tags=['system'])
def model_info():
    p = predictor()
    m = p.meta
    return {'trained_at': m['trained_at'], 'training_rows': m['training_rows'], 'crop_router': m['crop_router'],
            'crops': m['crops'], 'conditions': m['conditions']}


# ---------------------------------------------------------------- auth
@app.post('/api/auth/register', tags=['auth'], status_code=201)
def register(body: RegisterIn):
    email = body.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(422, 'Please enter a valid email address.')
    if db.get_user_by_email(email):
        raise HTTPException(409, 'An account with this email already exists.')
    user = db.create_user(body.name.strip(), email, auth.hash_password(body.password))
    return auth_response(user)


@app.post('/api/auth/login', tags=['auth'])
def login(body: LoginIn):
    user = db.get_user_by_email(body.email.strip().lower())
    if not user or not auth.verify_password(body.password, user['password_hash']):
        raise HTTPException(401, 'Wrong email or password.')
    return auth_response(user)


@app.get('/api/auth/me', tags=['auth'])
def me(user: dict = Depends(current_user)):
    return user


# ---------------------------------------------------------------- diagnosis
@app.post('/api/predict', tags=['diagnosis'], status_code=201)
async def predict(request: Request, crop: str | None = Query(default=None, description='Optional known crop code'),
                  x_filename: str | None = Header(default=None), user: dict = Depends(current_user)):
    """Diagnose one leaf photo. Send the image file as the raw request body (Content-Type: image/*)."""
    body = await request.body()
    if not body:
        raise HTTPException(400, 'No image received.')
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, 'The image is larger than 25 MB.')
    model = predictor()
    try:
        # CPU-heavy work runs in a worker thread so other requests are still answered meanwhile
        result, images, original = await run_in_threadpool(model.predict, body, crop or None)
    except ImageError as e:
        raise HTTPException(422, str(e)) from e

    pid = uuid.uuid4().hex
    image_file, overlay_file = f'{pid}.jpg', f'{pid}_leaf.jpg'
    step_files = {'exg': f'{pid}_exg.jpg', 'mask': f'{pid}_mask.png', 'regions': f'{pid}_regions.jpg',
                  'spots': f'{pid}_spots.jpg'}
    await run_in_threadpool(save_images, original, images, image_file, overlay_file, step_files)
    result['trace']['images'] = {k: f'/files/{v}' for k, v in step_files.items()}
    filename = unquote(x_filename)[:200] if x_filename else 'leaf.jpg'
    return db.add_prediction(pid, user['id'], filename, image_file, overlay_file, result)


def save_images(original, images: dict, image_file: str, overlay_file: str, step_files: dict):
    jpg = [cv2.IMWRITE_JPEG_QUALITY, 88]
    cv2.imwrite(os.path.join(db.UPLOAD_DIR, image_file), Predictor.resize(original, 1200), jpg)
    cv2.imwrite(os.path.join(db.UPLOAD_DIR, overlay_file), images['overlay'], jpg)
    for key, name in step_files.items():
        cv2.imwrite(os.path.join(db.UPLOAD_DIR, name), images[key], jpg if name.endswith('.jpg') else [])


@app.get('/api/predictions', tags=['history'])
def list_predictions(user: dict = Depends(current_user)):
    return db.list_predictions(user['id'])


@app.get('/api/predictions/stats', tags=['history'])
def prediction_stats(user: dict = Depends(current_user)):
    return db.stats(user['id'])


@app.get('/api/predictions/{pid}', tags=['history'])
def get_prediction(pid: str, user: dict = Depends(current_user)):
    rec = db.get_prediction(user['id'], pid)
    if not rec:
        raise HTTPException(404, 'Diagnosis not found.')
    return rec


@app.patch('/api/predictions/{pid}', tags=['history'])
def update_prediction(pid: str, body: NoteIn, user: dict = Depends(current_user)):
    rec = db.update_note(user['id'], pid, body.note.strip())
    if not rec:
        raise HTTPException(404, 'Diagnosis not found.')
    return rec


@app.delete('/api/predictions/{pid}', tags=['history'], status_code=204)
def delete_prediction(pid: str, user: dict = Depends(current_user)):
    rec = db.delete_prediction(user['id'], pid)
    if not rec:
        raise HTTPException(404, 'Diagnosis not found.')
    # the photo, the leaf outline and the step pictures all start with the diagnosis id
    for path in glob.glob(os.path.join(db.UPLOAD_DIR, f'{rec["id"]}*')):
        os.remove(path)
